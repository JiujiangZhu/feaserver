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
#ifndef CUFALLOC_H
#define CUFALLOC_H

/*
 *	This is the header file supporting cuFalloc.cu and defining both the host and device-side interfaces. See that file for some more
 *	explanation and sample use code. See also below for details of the host-side interfaces.
 *
 *  Quick sample code:
 *
	#include "cuFalloc.cu"
 	
	__global__ void TestFalloc(fallocDeviceHeap* deviceHeap)
	{
		fallocInit(deviceHeap);

		// create/free heap
		void* obj = fallocGetChunk(deviceHeap);
		fallocFreeChunk(deviceHeap, obj);

		// create/free alloc
		fallocContext* ctx = fallocCreateCtx(deviceHeap);
		char* testString = (char* )falloc(ctx, 10);
		int* testInteger = (int* )falloc(ctx, sizeof(int));
		fallocDisposeCtx(ctx);
	}

	int main()
	{
		cudaFallocHeap heap = cudaFallocInit(1);

		// test
		TestFalloc<<<1, 1>>>(heap.deviceHeap);

		// free and exit
		cudaFallocEnd(heap);
		printf("\ndone.\n"); // scanf("%c");
		return 0;
	}
 */


///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code

typedef struct _cuFallocDeviceHeap fallocDeviceHeap;
__device__ void fallocInit(fallocDeviceHeap* deviceHeap);
__device__ void* fallocGetChunk(fallocDeviceHeap* deviceHeap);
__device__ void fallocFreeChunk(fallocDeviceHeap* deviceHeap, void* obj);
// ALLOC
typedef struct _cuFallocContext fallocContext;
__device__ fallocContext* fallocCreateCtx(fallocDeviceHeap* deviceHeap);
__device__ void fallocDisposeCtx(fallocContext* ctx);
__device__ void* falloc(fallocContext* ctx, unsigned short bytes);


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

typedef struct {
	fallocDeviceHeap* deviceHeap;
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
extern "C" cudaFallocHeap cudaFallocInit(size_t bufferLen=1048576, cudaError_t* error=nullptr);   // 1-meg

//
//	cudaFallocEnd
//
//	Cleans up all memories allocated by cudaFallocInit() for a heap.
//	Call this at exit, or before calling cudaFallocInit() again.
//
extern "C" void cudaFallocEnd(cudaFallocHeap &heap);


#endif // CUFALLOC_H