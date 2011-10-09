#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
#ifndef CUFALLOC_C
#define CUFALLOC_C

#include "cuFalloc.cuh"
//#include <sm_11_atomic_functions.h>

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single printf() can take up
const static int HEAPCHUNK_SIZE = 128;
const static int FALLOCNODE_SLACK = 0x10;

// This is the header preceeding all printf entries.
// NOTE: It *must* be size-aligned to the maximum entity size (size_t)
typedef struct __align__(8) _cuFallocHeapChunk {
    unsigned short magic;				// Magic number says we're valid
    volatile struct _cuFallocHeapChunk* next;	// Next chunk pointer
} cuFallocHeapChunk;

typedef struct __align__(8) _cuFallocDeviceHeap {
	unsigned short chunks;
	volatile cuFallocHeapChunk* freeChunks;
} fallocDeviceHeap;

typedef struct _cuFallocDeviceNode {
	struct _cuFallocDeviceNode* next;
	struct _cuFallocDeviceNode* nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} cuFallocDeviceNode;

typedef struct _cuFallocContext {
	cuFallocDeviceNode node;
	cuFallocDeviceNode* nodes;
	cuFallocDeviceNode* availableNodes;
	fallocDeviceHeap* deviceHeap;
} fallocContext;

// All our headers are prefixed with a magic number so we know they're ready
#define CUFALLOC_MAGIC (unsigned short)0x3412        // Not a valid ascii character
#define CUFALLOCNODE_MAGIC (unsigned short)0x7856

__device__ void fallocInit(fallocDeviceHeap* deviceHeap) {
	if (threadIdx.x != 0)
		return;
	volatile cuFallocHeapChunk* chunk = (cuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap));
	deviceHeap->freeChunks = chunk;
	unsigned short chunks = deviceHeap->chunks;
	// preset all chunks
	chunk->magic = CUFALLOC_MAGIC;
	while (chunks-- > 1)
	{
		chunk = chunk->next = (cuFallocHeapChunk*)((__int8*)chunk + sizeof(cuFallocHeapChunk) + HEAPCHUNK_SIZE);
		chunk->magic = CUFALLOC_MAGIC;
	}
	chunk->next = nullptr;
	chunk->magic = CUFALLOC_MAGIC;
}

__device__ void* fallocGetChunk(fallocDeviceHeap* deviceHeap) {
	if (threadIdx.x != 0)
		__THROW;
	volatile cuFallocHeapChunk* chunk = deviceHeap->freeChunks;
	if (chunk == nullptr)
		return nullptr;
	{ // critical
		deviceHeap->freeChunks = chunk->next;
		chunk->next = nullptr;
	}
	return (void*)((short*)chunk + sizeof(cuFallocHeapChunk));
}

__device__ void fallocFreeChunk(fallocDeviceHeap* deviceHeap, void* obj) {
	if (threadIdx.x != 0)
		__THROW;
	cuFallocHeapChunk* chunk = (cuFallocHeapChunk*)((__int8*)obj - sizeof(cuFallocHeapChunk));
	if (chunk->magic != CUFALLOC_MAGIC)
		__THROW;
	{ // critical
		chunk->next = deviceHeap->freeChunks;
		deviceHeap->freeChunks = chunk;
	}
}

//////////////////////
// ALLOC

__device__ static fallocContext* fallocCreateCtx(fallocDeviceHeap* deviceHeap) {
	if (sizeof(fallocContext) > HEAPCHUNK_SIZE)
		__THROW;
	fallocContext* ctx = (fallocContext*)fallocGetChunk(deviceHeap);
	if (ctx == nullptr)
		__THROW;
	ctx->deviceHeap = deviceHeap;
	unsigned short freeOffset = ctx->node.freeOffset = sizeof(fallocContext);
	ctx->node.magic = CUFALLOCNODE_MAGIC;
	ctx->node.next = nullptr; ctx->nodes = (cuFallocDeviceNode*)ctx;
	ctx->node.nextAvailable = nullptr; ctx->availableNodes = (cuFallocDeviceNode*)ctx;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)
		ctx->availableNodes = nullptr;
	return ctx;
}

__device__ static void fallocDisposeCtx(fallocContext* ctx) {
	fallocDeviceHeap* deviceHeap = ctx->deviceHeap;
	for (cuFallocDeviceNode* node = ctx->nodes; node != nullptr; node = node->next)
		fallocFreeChunk(deviceHeap, node);
}

__device__ static void* falloc(fallocContext* ctx, unsigned short bytes, bool alloc) {
	if (bytes > (HEAPCHUNK_SIZE - sizeof(fallocContext)))
		__THROW;
	// find or add available node
	cuFallocDeviceNode* node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	cuFallocDeviceNode* lastNode;
	for (lastNode = (cuFallocDeviceNode*)ctx, node = ctx->availableNodes; node != nullptr; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		 if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			 break;
	if ((node == nullptr) || !hasFreeSpace) {
		// add node
		node = (cuFallocDeviceNode*)fallocGetChunk(ctx->deviceHeap);
		if (node == nullptr)
			__THROW;
		freeOffset = node->freeOffset = sizeof(cuFallocDeviceNode); 
		freeOffset += bytes;
		node->magic = CUFALLOCNODE_MAGIC;
		node->next = ctx->nodes; ctx->nodes = node;
		node->nextAvailable = (alloc ? ctx->availableNodes : nullptr); ctx->availableNodes = node;
	}
	//
	void* obj = (__int8*)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if (alloc && ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)) {
		if (lastNode == (cuFallocDeviceNode*)ctx)
			ctx->availableNodes = node->nextAvailable;
		else
			lastNode->nextAvailable = node->nextAvailable;
		node->nextAvailable = nullptr;
	}
	return obj;
}

__device__ static void* fallocRetract(fallocContext* ctx, unsigned short bytes) {
	cuFallocDeviceNode* node = ctx->availableNodes;
	int freeOffset = (int)node->freeOffset - bytes;
	// multi node, retract node
	if ((node != &ctx->node) && (freeOffset < sizeof(cuFallocDeviceNode)))
	{
		node->freeOffset = sizeof(cuFallocDeviceNode);
		// search for previous node
		cuFallocDeviceNode* lastNode;
		for (lastNode = (cuFallocDeviceNode*)ctx, node = ctx->nodes; node != nullptr; lastNode = node, node = node->next)
			if (node == ctx->availableNodes)
				break;
		node = ctx->availableNodes = lastNode;
		freeOffset = (int)node->freeOffset - bytes;
	}
	// first node && !overflow
	if ((node == &ctx->node) && (freeOffset < sizeof(fallocContext)))
		__THROW;
	node->freeOffset = (unsigned short)freeOffset;
	return (__int8*)node + freeOffset;
}

__device__ static void fallocMark(fallocContext* ctx, void* &mark, unsigned short &mark2) { mark = ctx->availableNodes; mark2 = ctx->availableNodes->freeOffset; }
__device__ static bool fallocAtMark(fallocContext* ctx, void* mark, unsigned short mark2) { return ((mark == ctx->availableNodes) && (mark2 == ctx->availableNodes->freeOffset)); }


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cudaFallocInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cudaFallocHeap cudaFallocInit(size_t bufferLen, cudaError_t* error) {
	cudaFallocHeap heap; memset(&heap, 0, sizeof(cudaFallocHeap));
	// Fix up chunkSize to include cpuFallocHeapChunk
	int chunkSize = sizeof(cuFallocHeapChunk) + HEAPCHUNK_SIZE;
	if ((chunkSize % 16) > 0)
        chunkSize += (16 - (chunkSize % 16));
    // Fix up bufferlen to be a multiple of chunkSize
    bufferLen = (bufferLen < chunkSize ? chunkSize : bufferLen);
    if ((bufferLen % chunkSize) > 0)
        bufferLen += (chunkSize - (bufferLen % chunkSize));
	unsigned short chunks = bufferLen / chunkSize;
	// Fix up bufferlen to include cudaFallocHeap
	bufferLen += sizeof(cudaFallocHeap);
	if ((bufferLen % 16) > 0)
        bufferLen += (16 - (bufferLen % 16));
    // Allocate a print buffer on the device and zero it
	fallocDeviceHeap* deviceHeap;
	if ( ((error == nullptr) && (cudaMalloc((void**)&deviceHeap, bufferLen) != cudaSuccess)) ||
		((error != nullptr) && ((*error = cudaMalloc((void**)&deviceHeap, bufferLen)) != cudaSuccess)) )
		return heap;
    cudaMemset(deviceHeap, 0, bufferLen);
	// transfer to deviceHeap
	fallocDeviceHeap hostHeap;
	hostHeap.freeChunks = nullptr;
	hostHeap.chunks = chunks;
	cudaMemcpy(deviceHeap, &hostHeap, sizeof(fallocDeviceHeap), cudaMemcpyHostToDevice);
	// return deviceHeap
	if (error != nullptr)
		*error = cudaSuccess;
	heap.deviceHeap = deviceHeap;
	heap.length = (int)bufferLen;
    return heap;
}

//
//  cudaFallocEnd
//
//  Frees up the memory which we allocated
//
extern "C" void cudaFallocEnd(cudaFallocHeap &heap) {
    if (!heap.deviceHeap)
        return;
    cudaFree(heap.deviceHeap); heap.deviceHeap = nullptr;
}


#endif // CUFALLOC_C