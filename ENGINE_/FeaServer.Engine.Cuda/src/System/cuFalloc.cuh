#ifndef CUFALLOC_H
#define CUFALLOC_H

/*
 *	This is the header file supporting cuFalloc.cu and defining both the host and device-side interfaces. See that file for some more
 *	explanation and sample use code. See also below for details of the host-side interfaces.
 *
 *  Quick sample code:
 *
	#include "cuFalloc.cu"
 	
	__global__ void testKernel(int val)
	{
		cuFallocHeap(12);
	}

	int main()
	{
		cudaFallocInit();
		testKernel<<< 2, 3 >>>(10);
		cudaFallocEnd();
        return 0;
	}
 */


///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

typedef struct _cuFallocDeviceHeap fallocDeviceHeap;
__device__ void fallocInit(fallocDeviceHeap *deviceHeap);
__device__ void *fallocGetChunk(fallocDeviceHeap *deviceHeap);
__device__ void fallocFreeChunk(fallocDeviceHeap *deviceHeap, void *obj);
// ALLOC
typedef struct _cuFallocDeviceContext fallocDeviceContext;
__device__ fallocDeviceContext *fallocCreate(fallocDeviceHeap *deviceHeap);
__device__ void fallocDispose(fallocDeviceContext *t);
__device__ void *falloc(fallocDeviceContext *t, unsigned short bytes);


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

typedef struct {
	void *deviceHeap;
	int length;
} cudaFallocHeap;

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
extern "C" cudaFallocHeap cudaFallocInit(size_t bufferLen=1048576, cudaError_t *error=0);   // 1-meg

//
//	cudaFallocEnd
//
//	Cleans up all memories allocated by cudaFallocInit() for a heap.
//	Call this at exit, or before calling cudaFallocInit() again.
//
extern "C" void cudaFallocEnd(cudaFallocHeap &heap);


#endif // CUFALLOC_H