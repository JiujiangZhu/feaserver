#ifndef CUFALLOC_C
#define CUFALLOC_C

#include "cuFalloc.cuh"
//#include <sm_11_atomic_functions.h>

// This is the smallest amount of memory, per-thread, which is allowed.
// It is also the largest amount of space a single printf() can take up
const static int HEAPCHUNK_SIZE = 256;
const static int FALLOCNODE_SLACK = 0x10;

// This is the header preceeding all printf entries.
// NOTE: It *must* be size-aligned to the maximum entity size (size_t)
typedef struct __align__(8) _cuFallocHeapChunk {
    unsigned short magic;				// Magic number says we're valid
    volatile struct _cuFallocHeapChunk *next;	// Next chunk pointer
} cuFallocHeapChunk;

typedef struct __align__(8) _cuFallocDeviceHeap {
	unsigned short chunks;
	volatile cuFallocHeapChunk* freeChunks;
} fallocDeviceHeap;

typedef struct _cuFallocDeviceNode {
	struct _cuFallocDeviceNode *next;
	struct _cuFallocDeviceNode *nextAvailable;
	unsigned short freeOffset;
	unsigned short magic;
} cuFallocDeviceNode;

typedef struct _cuFallocDeviceContext {
	cuFallocDeviceNode node;
	cuFallocDeviceNode *allocNodes;
	cuFallocDeviceNode *availableNodes;
	fallocDeviceHeap *deviceHeap;
} fallocDeviceContext;

// All our headers are prefixed with a magic number so we know they're ready
#define CUFALLOC_MAGIC (unsigned short)0x3412        // Not a valid ascii character
#define CUFALLOCNODE_MAGIC (unsigned short)0x7856
#define CUFALLOCNODE_SIZE (HEAPCHUNK_SIZE - sizeof(cuFallocDeviceNode))

__device__ void fallocInit(fallocDeviceHeap *deviceHeap) {
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

__device__ void *fallocGetChunk(fallocDeviceHeap *deviceHeap) {
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

__device__ void fallocFreeChunk(fallocDeviceHeap *deviceHeap, void *obj) {
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

__device__ static fallocDeviceContext *fallocCreate(fallocDeviceHeap *deviceHeap) {
	if (sizeof(fallocDeviceContext) > HEAPCHUNK_SIZE)
		__THROW;
	fallocDeviceContext *context = (fallocDeviceContext*)fallocGetChunk(deviceHeap);
	context->deviceHeap = deviceHeap;
	context->node.next = context->node.nextAvailable = nullptr;
	unsigned short freeOffset = context->node.freeOffset = sizeof(fallocDeviceContext);
	context->node.magic = CUFALLOCNODE_MAGIC;
	context->allocNodes = (cuFallocDeviceNode*)context;
	context->availableNodes = (cuFallocDeviceNode*)context;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE)
		context->availableNodes = nullptr;
	return context;
}

__device__ static void fallocDispose(fallocDeviceContext *t) {
	fallocDeviceHeap *deviceHeap = t->deviceHeap;
	for (cuFallocDeviceNode* node = t->allocNodes; node != nullptr; node = node->next)
		fallocFreeChunk(deviceHeap, node);
}

__device__ static void *falloc(fallocDeviceContext* t, unsigned short bytes) {
	if (bytes > CUFALLOCNODE_SIZE)
		__THROW;
	// find or add available node
	cuFallocDeviceNode *node;
	unsigned short freeOffset;
	unsigned char hasFreeSpace;
	cuFallocDeviceNode *lastNode;
	for (lastNode = (cuFallocDeviceNode*)t, node = t->availableNodes; node != nullptr; lastNode = node, node = node->nextAvailable)
		 if (hasFreeSpace = ((freeOffset = (node->freeOffset + bytes)) <= HEAPCHUNK_SIZE))
			 break;
	if ((node == nullptr) || !hasFreeSpace) {
		// add node
		node = (cuFallocDeviceNode*)fallocGetChunk(t->deviceHeap);
		node->next = t->allocNodes;
		node->nextAvailable = t->availableNodes;
		freeOffset = node->freeOffset = sizeof(cuFallocDeviceNode); 
		node->magic = CUFALLOCNODE_MAGIC;
		t->allocNodes = node;
		t->availableNodes = node;
	}
	void *obj = (__int8*)node + node->freeOffset;
	node->freeOffset = freeOffset;
	// close node
	if ((freeOffset + FALLOCNODE_SLACK) > HEAPCHUNK_SIZE) {
		if (lastNode == (cuFallocDeviceNode*)t)
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
extern "C" cudaFallocHeap cudaFallocInit(size_t bufferLen, cudaError_t *error) {
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
	fallocDeviceHeap *deviceHeap;
	if ( ((error == nullptr) && (cudaMalloc((void **)&deviceHeap, bufferLen) != cudaSuccess)) ||
		((error != nullptr) && ((*error = cudaMalloc((void **)&deviceHeap, bufferLen)) != cudaSuccess)) )
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