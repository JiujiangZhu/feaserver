#ifndef CPUFALLOC_C
#define CPUFALLOC_C

#include <stdlib.h>
#include <string.h>
#include "cpuFalloc.h"

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single falloc() can take up
const static int HEAPCHUNK_SIZE = 256;
const static int FALLOCNODE_SLACK = 0x10;

// This is the header preceeding all printf entries.
// NOTE: It *must* be size-aligned to the maximum entity size (size_t)
typedef struct _cpuFallocHeapChunk {
    unsigned short magic;				// Magic number says we're valid
    volatile struct _cpuFallocHeapChunk *next;	// Next chunk pointer
} cpuFallocHeapChunk;

typedef struct _cpuFallocDeviceHeap {
	unsigned short chunks;
	volatile cpuFallocHeapChunk* freeChunks;
} fallocDeviceHeap;

typedef struct _cpuFallocDeviceNode {
	struct _cpuFallocDeviceNode *next;
	struct _cpuFallocDeviceNode *nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} cpuFallocDeviceNode;

typedef struct _cpuFallocDeviceContext {
	cpuFallocDeviceNode node;
	cpuFallocDeviceNode *allocNodes;
	cpuFallocDeviceNode *availableNodes;
	fallocDeviceHeap *deviceHeap;
} fallocDeviceContext;

// All our headers are prefixed with a magic number so we know they're ready
#define CPUFALLOC_MAGIC (unsigned short)0x3412        // Not a valid ascii character
#define CPUFALLOCNODE_MAGIC (unsigned short)0x7856
#define CPUFALLOCNODE_SIZE (HEAPCHUNK_SIZE - sizeof(cpuFallocDeviceNode))

void fallocInit(fallocDeviceHeap *deviceHeap) {
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

void *fallocGetChunk(fallocDeviceHeap *deviceHeap) {
	volatile cpuFallocHeapChunk* chunk = deviceHeap->freeChunks;
	if (chunk == nullptr)
		return nullptr;
	{ // critical
		deviceHeap->freeChunks = chunk->next;
		chunk->next = nullptr;
	}
	return (void*)((__int8*)chunk + sizeof(cpuFallocHeapChunk));
}

void fallocFreeChunk(fallocDeviceHeap *deviceHeap, void *obj) {
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

fallocDeviceContext *fallocCreateCtx(fallocDeviceHeap *deviceHeap) {
	if (sizeof(fallocDeviceContext) > HEAPCHUNK_SIZE)
		throw;
	fallocDeviceContext *context = (fallocDeviceContext*)fallocGetChunk(deviceHeap);
	context->deviceHeap = deviceHeap;
	context->node.next = context->node.nextAvailable = nullptr;
	unsigned short freeOffset = context->node.freeOffset = sizeof(fallocDeviceContext);
	context->node.magic = CPUFALLOCNODE_MAGIC;
	context->allocNodes = (cpuFallocDeviceNode*)context;
	context->availableNodes = (cpuFallocDeviceNode*)context;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)
		context->availableNodes = nullptr;
	return context;
}

void fallocDisposeCtx(fallocDeviceContext *t) {
	fallocDeviceHeap *deviceHeap = t->deviceHeap;
	for (cpuFallocDeviceNode* node = t->allocNodes; node != nullptr; node = node->next)
		fallocFreeChunk(deviceHeap, node);
}

void *falloc(fallocDeviceContext* t, unsigned short bytes) {
	if (bytes > CPUFALLOCNODE_SIZE)
		throw;
	// find or add available node
	cpuFallocDeviceNode *node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	cpuFallocDeviceNode *lastNode;
	for (lastNode = (cpuFallocDeviceNode*)t, node = t->availableNodes; node != nullptr; lastNode = node, node = node->nextAvailable)
		 if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			 break;
	if ((node == nullptr) || !hasFreeSpace) {
		// add node
		node = (cpuFallocDeviceNode*)fallocGetChunk(t->deviceHeap);
		node->next = t->allocNodes;
		node->nextAvailable = t->availableNodes;
		freeOffset = node->freeOffset = sizeof(cpuFallocDeviceNode); 
		node->magic = CPUFALLOCNODE_MAGIC;
		t->allocNodes = node;
		t->availableNodes = node;
	}
	void *obj = (__int8*)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE) {
		if (lastNode == (cpuFallocDeviceNode*)t)
			t->availableNodes = node->nextAvailable;
		else
			lastNode->nextAvailable = node->nextAvailable;
		node->nextAvailable = nullptr;
	}
	return obj;
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
	fallocDeviceHeap *deviceHeap;
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