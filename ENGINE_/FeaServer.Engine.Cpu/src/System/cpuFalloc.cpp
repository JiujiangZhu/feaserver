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
#ifndef CPUFALLOC_C
#define CPUFALLOC_C

#include <stdlib.h>
#include <string.h>
#include "cpuFalloc.h"

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single falloc() can take up
const static int HEAPCHUNK_SIZE = 128; //256;
const static int FALLOCNODE_SLACK = 0x10;

// This is the header preceeding all printf entries.
// NOTE: It *must* be size-aligned to the maximum entity size (size_t)
typedef struct _cpuFallocHeapChunk {
    unsigned short magic;				// Magic number says we're valid
    volatile struct _cpuFallocHeapChunk* next;	// Next chunk pointer
} cpuFallocHeapChunk;

typedef struct _cpuFallocDeviceHeap {
	unsigned short chunks;
	volatile cpuFallocHeapChunk* freeChunks;
} fallocDeviceHeap;

typedef struct _cpuFallocDeviceNode {
	struct _cpuFallocDeviceNode* next;
	struct _cpuFallocDeviceNode* nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} cpuFallocDeviceNode;

typedef struct _cpuFallocContext {
	cpuFallocDeviceNode node;
	cpuFallocDeviceNode* nodes;
	cpuFallocDeviceNode* availableNodes;
	fallocDeviceHeap* deviceHeap;
} fallocContext;

// All our headers are prefixed with a magic number so we know they're ready
#define CPUFALLOC_MAGIC (unsigned short)0x3412        // Not a valid ascii character
#define CPUFALLOCNODE_MAGIC (unsigned short)0x7856

void fallocInit(fallocDeviceHeap* deviceHeap) {
	volatile cpuFallocHeapChunk* chunk = (cpuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap));
	deviceHeap->freeChunks = chunk;
	unsigned short chunks = deviceHeap->chunks;
	// preset all chunks
	chunk->magic = CPUFALLOC_MAGIC;
	while (chunks-- > 1)
	{
		chunk = chunk->next = (cpuFallocHeapChunk*)((__int8*)chunk + sizeof(cpuFallocHeapChunk) + HEAPCHUNK_SIZE);
		chunk->magic = CPUFALLOC_MAGIC;
	}
	chunk->next = nullptr;
	chunk->magic = CPUFALLOC_MAGIC;
}

void* fallocGetChunk(fallocDeviceHeap* deviceHeap) {
	volatile cpuFallocHeapChunk* chunk = deviceHeap->freeChunks;
	if (chunk == nullptr)
		return nullptr;
	{ // critical
		deviceHeap->freeChunks = chunk->next;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(cpuFallocHeapChunk));
}

void fallocFreeChunk(fallocDeviceHeap* deviceHeap, void* obj) {
	cpuFallocHeapChunk* chunk = (cpuFallocHeapChunk*)((__int8*)obj - sizeof(cpuFallocHeapChunk));
	if (chunk->magic != CPUFALLOC_MAGIC)
		throw;
	{ // critical
		chunk->next = deviceHeap->freeChunks;
		deviceHeap->freeChunks = chunk;
	}
}

//////////////////////
// ALLOC

fallocContext* fallocCreateCtx(fallocDeviceHeap* deviceHeap) {
	if (sizeof(fallocContext) > HEAPCHUNK_SIZE)
		throw;
	fallocContext* ctx = (fallocContext*)fallocGetChunk(deviceHeap);
	if (ctx == nullptr)
		throw;
	ctx->deviceHeap = deviceHeap;
	unsigned short freeOffset = ctx->node.freeOffset = sizeof(fallocContext);
	ctx->node.magic = CPUFALLOCNODE_MAGIC;
	ctx->node.next = nullptr; ctx->nodes = (cpuFallocDeviceNode*)ctx;
	ctx->node.nextAvailable = nullptr; ctx->availableNodes = (cpuFallocDeviceNode*)ctx;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)
		ctx->availableNodes = nullptr;
	return ctx;
}

void fallocDisposeCtx(fallocContext* ctx) {
	fallocDeviceHeap* deviceHeap = ctx->deviceHeap;
	for (cpuFallocDeviceNode* node = ctx->nodes; node != nullptr; node = node->next)
		fallocFreeChunk(deviceHeap, node);
}

void* falloc(fallocContext* ctx, unsigned short bytes, bool alloc) {
	if (bytes > (HEAPCHUNK_SIZE - sizeof(fallocContext)))
		throw;
	// find or add available node
	cpuFallocDeviceNode* node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	cpuFallocDeviceNode* lastNode;
	for (lastNode = (cpuFallocDeviceNode*)ctx, node = ctx->availableNodes; node != nullptr; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			break;
	if ((node == nullptr) || !hasFreeSpace) {
		// add node
		node = (cpuFallocDeviceNode*)fallocGetChunk(ctx->deviceHeap);
		if (node == nullptr)
			throw;
		freeOffset = node->freeOffset = sizeof(cpuFallocDeviceNode);
		freeOffset += bytes;
		node->magic = CPUFALLOCNODE_MAGIC;
		node->next = ctx->nodes; ctx->nodes = node;
		node->nextAvailable = (alloc ? ctx->availableNodes : nullptr); ctx->availableNodes = node;
	}
	//
	void* obj = (__int8*)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if (alloc && ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)) {
		if (lastNode == (cpuFallocDeviceNode*)ctx)
			ctx->availableNodes = node->nextAvailable;
		else
			lastNode->nextAvailable = node->nextAvailable;
		node->nextAvailable = nullptr;
	}
	return obj;
}

void* fallocRetract(fallocContext* ctx, unsigned short bytes) {
	cpuFallocDeviceNode* node = ctx->availableNodes;
	int freeOffset = (int)node->freeOffset - bytes;
	// multi node, retract node
	if ((node != &ctx->node) && (freeOffset < sizeof(cpuFallocDeviceNode)))
	{
		node->freeOffset = sizeof(cpuFallocDeviceNode);
		// search for previous node
		cpuFallocDeviceNode* lastNode;
		for (lastNode = (cpuFallocDeviceNode*)ctx, node = ctx->nodes; node != nullptr; lastNode = node, node = node->next)
			if (node == ctx->availableNodes)
				break;
		node = ctx->availableNodes = lastNode;
		freeOffset = (int)node->freeOffset - bytes;
	}
	// first node && !overflow
	if ((node == &ctx->node) && (freeOffset < sizeof(fallocContext)))
		throw;
	node->freeOffset = (unsigned short)freeOffset;
	return (__int8*)node + freeOffset;
}

void fallocMark(fallocContext* ctx, void* &mark, unsigned short &mark2) { mark = ctx->availableNodes; mark2 = ctx->availableNodes->freeOffset; }
bool fallocAtMark(fallocContext* ctx, void* mark, unsigned short mark2) { return ((mark == ctx->availableNodes) && (mark2 == ctx->availableNodes->freeOffset)); }


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cudaFallocInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cpuFallocHeap cpuFallocInit(size_t bufferLen) {
	cpuFallocHeap heap; memset(&heap, 0, sizeof(fallocDeviceHeap));
	// Fix up chunkSize to include cpuFallocHeapChunk
	int chunkSize = sizeof(cpuFallocHeapChunk) + HEAPCHUNK_SIZE;
	if ((chunkSize % 16) > 0)
        chunkSize += (16 - (chunkSize % 16));
    // Fix up bufferlen to be a multiple of chunkSize
    bufferLen = (bufferLen < chunkSize ? chunkSize : bufferLen);
    if ((bufferLen % chunkSize) > 0)
        bufferLen += (chunkSize - (bufferLen % chunkSize));
	unsigned short chunks = (unsigned short)(bufferLen / chunkSize);
	// Fix up bufferlen to include fallocDeviceHeap
	bufferLen += sizeof(fallocDeviceHeap);
	if ((bufferLen % 16) > 0)
        bufferLen += (16 - (bufferLen % 16));
    // Allocate a print buffer on the device and zero it
	fallocDeviceHeap* deviceHeap;
    if ((deviceHeap = (fallocDeviceHeap*)malloc(bufferLen)) == nullptr)
		return heap;
    memset(deviceHeap, 0, bufferLen);
	// transfer to heap
	deviceHeap->chunks = chunks;
	// return heap
	heap.deviceHeap = deviceHeap;
	heap.length = (int)bufferLen;
    return heap;
}

//
//  cudaFallocEnd
//
//  Frees up the memory which we allocated
//
extern "C" void cpuFallocEnd(cpuFallocHeap &heap) {
	if (!heap.deviceHeap)
		return;
    free(heap.deviceHeap); heap.deviceHeap = nullptr;
}


#endif // CPUFALLOC_C