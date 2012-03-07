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

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single printf() can take up
const static int HEAPCHUNK_SIZE = 256;
const static int FALLOCNODE_SLACK = 0x10;

typedef struct __align__(8) _cuFallocHeapChunk {
    unsigned short magic;
	unsigned short count;
    volatile struct _cuFallocHeapChunk* next;
	void* reserved;
} cuFallocHeapChunk;

typedef struct __align__(8) _cuFallocDeviceHeap {
	size_t chunks;
	volatile cuFallocHeapChunk* freeChunks;
	void* reserved;
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
#define CHUNKSIZE (sizeof(cuFallocHeapChunk)+HEAPCHUNK_SIZE)
#define CHUNKSIZEALIGN (CHUNKSIZE%16?CHUNKSIZE+16-(CHUNKSIZE%16):CHUNKSIZE)

__device__ void fallocInit(fallocDeviceHeap* deviceHeap) {
	if ((threadIdx.x) || (threadIdx.y) || (threadIdx.z))
		return;
	volatile cuFallocHeapChunk* chunk = (cuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap));
	deviceHeap->freeChunks = chunk;
	size_t chunks = deviceHeap->chunks;
	if (!chunks)
		__THROW;
	// preset all chunks
	chunk->magic = CUFALLOC_MAGIC;
	chunk->count = 1;
	chunk->reserved = nullptr;
	while (chunks-- > 1) {
		chunk = chunk->next = (cuFallocHeapChunk*)((__int8*)chunk + CHUNKSIZEALIGN);
		chunk->magic = CUFALLOC_MAGIC;
		chunk->count = 1;
		chunk->reserved = nullptr;
	}
	chunk->next = nullptr;
}

__device__ void* fallocGetChunk(fallocDeviceHeap* deviceHeap) {
	if ((threadIdx.x) || (threadIdx.y) || (threadIdx.z))
		__THROW;
	volatile cuFallocHeapChunk* chunk = deviceHeap->freeChunks;
	if (!chunk)
		return nullptr;
	{ // critical
		deviceHeap->freeChunks = chunk->next;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(cuFallocHeapChunk));
}

__device__ void* fallocGetChunks(fallocDeviceHeap* deviceHeap, size_t length, size_t* allocLength) {
    // fix up length to be a multiple of chunkSize
    length = (length < CHUNKSIZEALIGN ? CHUNKSIZEALIGN : length);
    if (length % CHUNKSIZEALIGN)
        length += CHUNKSIZEALIGN - (length % CHUNKSIZEALIGN);
	// set length, if requested
	if (allocLength)
		*allocLength = length - sizeof(cuFallocHeapChunk);
	size_t chunks = (size_t)(length / CHUNKSIZEALIGN);
	if (chunks > deviceHeap->chunks)
		__THROW;
	// single, equals: fallocGetChunk
	if (chunks == 1)
		return fallocGetChunk(deviceHeap);
    // multiple, find a contiguous chuck
	size_t index = chunks;
	volatile cuFallocHeapChunk* chunk;
	volatile cuFallocHeapChunk* endChunk = (cuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap) + (CHUNKSIZEALIGN * deviceHeap->chunks));
	{ // critical
		for (chunk = (cuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap)); index && (chunk < endChunk); chunk = (cuFallocHeapChunk*)((__int8*)chunk + (CHUNKSIZEALIGN * chunk->count))) {
			if (chunk->magic != CUFALLOC_MAGIC)
				__THROW;
			index = (chunk->next ? index - 1 : chunks);
		}
		if (index)
			return nullptr;
		// found chuck, remove from freeChunks
		endChunk = chunk;
		chunk = (cuFallocHeapChunk*)((__int8*)chunk - (CHUNKSIZEALIGN * chunks));
		for (volatile cuFallocHeapChunk* chunk2 = deviceHeap->freeChunks; chunk2; chunk2 = chunk2->next)
			if ((chunk2 >= chunk) && (chunk2 <= endChunk))
				chunk2->next = (chunk2->next ? chunk2->next->next : nullptr);
		chunk->count = chunks;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(cuFallocHeapChunk));
}

__device__ void fallocFreeChunk(fallocDeviceHeap* deviceHeap, void* obj) {
	if ((threadIdx.x) || (threadIdx.y) || (threadIdx.z))
		__THROW;
	volatile cuFallocHeapChunk* chunk = (cuFallocHeapChunk*)((__int8*)obj - sizeof(cuFallocHeapChunk));
	if ((chunk->magic != CUFALLOC_MAGIC) || (chunk->count > 1))
		__THROW;
	{ // critical
		chunk->next = deviceHeap->freeChunks;
		deviceHeap->freeChunks = chunk;
	}
}

__device__ void fallocFreeChunks(fallocDeviceHeap* deviceHeap, void* obj) {
	volatile cuFallocHeapChunk* chunk = (cuFallocHeapChunk*)((__int8*)obj - sizeof(cuFallocHeapChunk));
	if (chunk->magic != CUFALLOC_MAGIC)
		__THROW;
	size_t chunks = chunk->count;
	// single, equals: fallocFreeChunk
	if (chunks == 1) {
		{ // critical
			chunk->next = deviceHeap->freeChunks;
			deviceHeap->freeChunks = chunk;
		}
		return;
	}
	// retag chunks
	chunk->count = 1;
	while (chunks-- > 1) {
		chunk = chunk->next = (cuFallocHeapChunk*)((__int8*)chunk + sizeof(cuFallocHeapChunk) + HEAPCHUNK_SIZE);
		chunk->magic = CUFALLOC_MAGIC;
		chunk->count = 1;
		chunk->reserved = nullptr;
	}
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
	if (!ctx)
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
	for (cuFallocDeviceNode* node = ctx->nodes; node; node = node->next)
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
	for (lastNode = (cuFallocDeviceNode*)ctx, node = ctx->availableNodes; node; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		 if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			 break;
	if (!node || !hasFreeSpace) {
		// add node
		node = (cuFallocDeviceNode*)fallocGetChunk(ctx->deviceHeap);
		if (!node)
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
	if ((node != &ctx->node) && (freeOffset < sizeof(cuFallocDeviceNode))) {
		node->freeOffset = sizeof(cuFallocDeviceNode);
		// search for previous node
		cuFallocDeviceNode* lastNode;
		for (lastNode = (cuFallocDeviceNode*)ctx, node = ctx->nodes; node; lastNode = node, node = node->next)
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


#if __CUDA_ARCH__ > 100 // atomics only used with > sm_10 architecture
//////////////////////
// ATOMIC
#include <sm_11_atomic_functions.h>

#define CPUFATOMIC_MAGIC (unsigned short)0xBC9A

typedef struct _cpuFallocAutomic {
	fallocDeviceHeap* deviceHeap;
	unsigned short magic;
	unsigned short pitch;
	size_t bufferLength;
	unsigned __int32* bufferBase;
	volatile unsigned __int32* buffer;
} fallocAutomic;

fallocAutomic* fallocCreateAtom(fallocDeviceHeap* deviceHeap, unsigned short pitch, size_t length) {
	// align pitch
	if (pitch % 16)
		pitch += 16 - (pitch % 16);
	// fix up length to be a multiple of pitch
    length = (length < pitch ? pitch : length);
    if (length % pitch)
        length += pitch - (length % pitch);
	//
	size_t allocLength;
	fallocAutomic* atom = (fallocAutomic*)fallocGetChunks(deviceHeap, length + sizeof(fallocAutomic), &allocLength);
	if (!atom)
		throw;
	atom->deviceHeap = deviceHeap;
	atom->magic = CPUFATOMIC_MAGIC;
	atom->pitch = pitch;
	atom->bufferLength = allocLength - sizeof(fallocAutomic);
	atom->bufferBase = (unsigned __int32*)atom + sizeof(fallocAutomic);
	atom->buffer = (volatile unsigned __int32*)atom->bufferBase;
	return atom;
}

void fallocDisposeAtom(fallocAutomic* atom) {
	fallocFreeChunks(atom->deviceHeap, atom);
}

void* fallocAtomNext(fallocAutomic* atom, unsigned short bytes) {
	size_t offset = atomicAdd(atom->bufferBase, atom->pitch) - (size_t)atom->buffer;
    offset %= atom->bufferLength;
    return (void*)(atom->buffer + offset);
}

#endif

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cudaFallocInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cudaFallocHeap cudaFallocInit(size_t length, cudaError_t* error, void* reserved) {
	cudaFallocHeap heap; memset(&heap, 0, sizeof(cudaFallocHeap));
    // fix up length to be a multiple of chunkSize
    length = (length < CHUNKSIZEALIGN ? CHUNKSIZEALIGN : length);
    if (length % CHUNKSIZEALIGN)
        length += (CHUNKSIZEALIGN - (length % CHUNKSIZEALIGN));
	size_t chunks = (size_t)(length / CHUNKSIZEALIGN);
	if (!chunks)
		return heap;
	// Fix up length to include cudaFallocHeap
	length += sizeof(cudaFallocHeap);
	if ((length % 16) > 0)
        length += 16 - (length % 16);
    // Allocate a print buffer on the device and zero it
	fallocDeviceHeap* deviceHeap;
	if ((!error && (cudaMalloc((void**)&deviceHeap, length) != cudaSuccess)) ||
		(error && ((*error = cudaMalloc((void**)&deviceHeap, length)) != cudaSuccess)))
		return heap;
    cudaMemset(deviceHeap, 0, length);
	// transfer to deviceHeap
	fallocDeviceHeap hostDeviceHeap;
	hostDeviceHeap.freeChunks = nullptr;
	hostDeviceHeap.chunks = chunks;
	hostDeviceHeap.reserved = reserved;
	if ((!error && (cudaMemcpy(deviceHeap, &hostDeviceHeap, sizeof(fallocDeviceHeap), cudaMemcpyHostToDevice) != cudaSuccess)) ||
		(error && ((*error = cudaMemcpy(deviceHeap, &hostDeviceHeap, sizeof(fallocDeviceHeap), cudaMemcpyHostToDevice)) != cudaSuccess)))
		return heap;
	// return the heap
	if (error)
		*error = cudaSuccess;
	heap.deviceHeap = deviceHeap;
	heap.length = (int)length;
	heap.reserved = reserved;
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