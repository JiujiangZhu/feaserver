#ifndef CPUFALLOC_H
#define CPUFALLOC_H

/*
 *	This is the header file supporting cpuFalloc.c and defining both the host and device-side interfaces. See that file for some more
 *	explanation and sample use code. See also below for details of the host-side interfaces.
 *
 *  Quick sample code:
 *
	#include "cpuFalloc.c"

	int main()
	{
		cpuFallocHeap heap = cpuFallocInit();
		fallocInit(heap.deviceHeap);

		// create/free heap
		void *obj = fallocGetChunk(heap.deviceHeap);
		fallocFreeChunk(heap.deviceHeap, obj);

		// create/free alloc
		fallocDeviceContext *ctx = fallocCreate(heap.deviceHeap);
		char *testString = (char *)falloc(ctx, 10);
		int *testInteger = (int *)falloc(ctx, sizeof(int));
		fallocDispose(ctx);

		// free and exit
		cpuFallocEnd(heap);
        return 0;
	}
 */


///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

typedef struct _cpuFallocDeviceHeap fallocDeviceHeap;
void fallocInit(fallocDeviceHeap *deviceHeap);
void *fallocGetChunk(fallocDeviceHeap *deviceHeap);
void fallocFreeChunk(fallocDeviceHeap *deviceHeap, void *obj);
// ALLOC
typedef struct _cpuFallocDeviceContext fallocDeviceContext;
fallocDeviceContext *fallocCreateCtx(fallocDeviceHeap *deviceHeap);
void fallocDisposeCtx(fallocDeviceContext *t);
void *falloc(fallocDeviceContext *t, unsigned short bytes);


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

typedef struct {
	fallocDeviceHeap *deviceHeap;
	int length;
} cpuFallocHeap;

//
//	cudaFallocInit
//
//	Call this to initialise a falloc heap. If the buffer size needs to be changed, call cudaFallocEnd()
//	before re-calling cudaFallocInit().
//
//	The default size for the buffer is 1 megabyte. For CUDA
//	architecture 1.1 and above, the buffer is filled linearly and
//	is completely used.
//
//	Arguments:
//		bufferLen - Length, in bytes, of total space to reserve (in device global memory) for output.
//
//	Returns:
//		cudaSuccess if all is well.
//
extern "C" cpuFallocHeap cpuFallocInit(size_t bufferLen=1048576);   // 1-meg

//
//	cudaFallocEnd
//
//	Cleans up all memories allocated by cudaFallocInit() for a heap.
//	Call this at exit, or before calling cudaFallocInit() again.
//
extern "C" void cpuFallocEnd(cpuFallocHeap &heap);


#endif // CPUFALLOC_H
