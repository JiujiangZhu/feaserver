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

typedef struct _cpuFallocHeapChunk {
    unsigned short magic; // magic number says we're valid
	unsigned short count; // count of chunks: default 1
    volatile struct _cpuFallocHeapChunk* next; // next chunk pointer
} cpuFallocHeapChunk;

typedef struct _cpuFallocDeviceHeap {
	unsigned short chunks;
	volatile cpuFallocHeapChunk* freeChunks;
} fallocDeviceHeap;

typedef struct _cpuFallocDeviceNode {
	struct _cpuFallocDeviceNode* next; // next node pointer
	struct _cpuFallocDeviceNode* nextAvailable; // next available-node pointer
	unsigned short freeOffset; // moving free-offset into node, starts from top and includes header
	unsigned short magic; // magic number says we're valid
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
#define CHUNKSIZE (sizeof(cpuFallocHeapChunk)+HEAPCHUNK_SIZE)
#define CHUNKSIZEALIGN (CHUNKSIZE%16?CHUNKSIZE+16-(CHUNKSIZE%16):CHUNKSIZE)

void fallocInit(fallocDeviceHeap* deviceHeap) {
	volatile cpuFallocHeapChunk* chunk = (cpuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap));
	deviceHeap->freeChunks = chunk;
	unsigned short chunks = deviceHeap->chunks;
	// preset all chunks
	chunk->magic = CPUFALLOC_MAGIC;
	chunk->count = 1;
	while (chunks-- > 1) {
		chunk = chunk->next = (cpuFallocHeapChunk*)((__int8*)chunk + CHUNKSIZEALIGN);
		chunk->magic = CPUFALLOC_MAGIC;
		chunk->count = 1;
	}
	chunk->next = nullptr;
}

void* fallocGetChunk(fallocDeviceHeap* deviceHeap) {
	volatile cpuFallocHeapChunk* chunk = deviceHeap->freeChunks;
	if (!chunk)
		return nullptr;
	{ // critical
		deviceHeap->freeChunks = chunk->next;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(cpuFallocHeapChunk));
}

void* fallocGetChunks(fallocDeviceHeap* deviceHeap, size_t length, size_t* allocLength) {
    // fix up length to be a multiple of chunkSize
    length = (length < CHUNKSIZEALIGN ? CHUNKSIZEALIGN : length);
    if (length % CHUNKSIZEALIGN)
        length += CHUNKSIZEALIGN - (length % CHUNKSIZEALIGN);
	// set length, if requested
	if (allocLength)
		*allocLength = length - sizeof(cpuFallocHeapChunk);
	unsigned short chunks = (unsigned short)(length / CHUNKSIZEALIGN);
	if (chunks > deviceHeap->chunks)
		throw;
	// single, equals: fallocGetChunk
	if (chunks == 1)
		return fallocGetChunk(deviceHeap);
    // multiple, find a contiguous chuck
	unsigned short index = chunks;
	volatile cpuFallocHeapChunk* chunk;
	volatile cpuFallocHeapChunk* endChunk = (cpuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap) + (CHUNKSIZEALIGN * chunks));
	{ // critical
		for (chunk = (cpuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap)); index && (chunk < endChunk); chunk = (cpuFallocHeapChunk*)((__int8*)chunk + (CHUNKSIZEALIGN * chunk->count))) {
			if (chunk->magic != CPUFALLOC_MAGIC)
				throw;
			index = (chunk->next ? index - 1 : chunks);
		}
		if (index)
			return nullptr;
		// found chuck, remove from freeChunks
		endChunk = chunk;
		chunk = (cpuFallocHeapChunk*)((__int8*)chunk - (CHUNKSIZEALIGN * chunks));
		for (volatile cpuFallocHeapChunk* chunk2 = deviceHeap->freeChunks; chunk2; chunk2 = chunk2->next)
			if ((chunk2 >= chunk) && (chunk2 <= endChunk))
				chunk2->next = (chunk2->next ? chunk2->next->next : nullptr);
		chunk->count = chunks;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(cpuFallocHeapChunk));
}

void fallocFreeChunk(fallocDeviceHeap* deviceHeap, void* obj) {
	volatile cpuFallocHeapChunk* chunk = (cpuFallocHeapChunk*)((__int8*)obj - sizeof(cpuFallocHeapChunk));
	if ((chunk->magic != CPUFALLOC_MAGIC) || (chunk->count > 1))
		throw;
	{ // critical
		chunk->next = deviceHeap->freeChunks;
		deviceHeap->freeChunks = chunk;
	}
}

void fallocFreeChunks(fallocDeviceHeap* deviceHeap, void* obj) {
	volatile cpuFallocHeapChunk* chunk = (cpuFallocHeapChunk*)((__int8*)obj - sizeof(cpuFallocHeapChunk));
	if (chunk->magic != CPUFALLOC_MAGIC)
		throw;
	unsigned short chunks = chunk->count;
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
		chunk = chunk->next = (cpuFallocHeapChunk*)((__int8*)chunk + sizeof(cpuFallocHeapChunk) + HEAPCHUNK_SIZE);
		chunk->magic = CPUFALLOC_MAGIC;
		chunk->count = 1;
	}
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
	if (!ctx)
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
	for (cpuFallocDeviceNode* node = ctx->nodes; node; node = node->next)
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
	for (lastNode = (cpuFallocDeviceNode*)ctx, node = ctx->availableNodes; node; lastNode = node, node = (alloc ? node->nextAvailable : node->next))
		if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			break;
	if (!node || !hasFreeSpace) {
		// add node
		node = (cpuFallocDeviceNode*)fallocGetChunk(ctx->deviceHeap);
		if (!node)
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
	if ((node != &ctx->node) && (freeOffset < sizeof(cpuFallocDeviceNode))) {
		node->freeOffset = sizeof(cpuFallocDeviceNode);
		// search for previous node
		cpuFallocDeviceNode* lastNode;
		for (lastNode = (cpuFallocDeviceNode*)ctx, node = ctx->nodes; node; lastNode = node, node = node->next)
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


//////////////////////
// ATOMIC

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
	size_t offset = (size_t)atom->buffer - (size_t)atom->bufferBase;
	atom->buffer += atom->pitch;
    offset %= atom->bufferLength;
	return atom->bufferBase + offset;
}


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cudaFallocInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cpuFallocHeap cpuFallocInit(size_t length) {
	cpuFallocHeap heap; memset(&heap, 0, sizeof(fallocDeviceHeap));
    // fix up length to be a multiple of chunkSize
    length = (length < CHUNKSIZEALIGN ? CHUNKSIZEALIGN : length);
    if (length % CHUNKSIZEALIGN)
        length += CHUNKSIZEALIGN - (length % CHUNKSIZEALIGN);
	unsigned short chunks = (unsigned short)(length / CHUNKSIZEALIGN);
	// fix up length to include fallocDeviceHeap
	length += sizeof(fallocDeviceHeap);
	if (length % 16)
        length += 16 - (length % 16);
    // allocate a print buffer on the device and zero it
	fallocDeviceHeap* deviceHeap;
    if (!(deviceHeap = (fallocDeviceHeap*)malloc(length)))
		return heap;
    memset(deviceHeap, 0, length);
	// transfer to heap
	deviceHeap->chunks = chunks;
	// return heap
	heap.deviceHeap = deviceHeap;
	heap.length = (int)length;
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