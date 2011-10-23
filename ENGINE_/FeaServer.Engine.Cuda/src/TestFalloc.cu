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
#pragma once

#include <cuda.h>;
#include "Core.h";
#include "System\cuFallocWTrace.cu"

__global__ void TestFalloc(fallocDeviceHeap* deviceHeap)
{
	fallocInit(deviceHeap);

	// create/free heap
	void* obj = fallocGetChunk(deviceHeap);
	fallocFreeChunk(deviceHeap, obj);
/*
	void* obj2 = fallocGetChunks(heap.deviceHeap, 144*2);
	fallocFreeChunks(heap.deviceHeap, obj2);
*/

	// create/free alloc
	fallocContext* ctx = fallocCreateCtx(deviceHeap);
	char* testString = (char*)falloc(ctx, 10);
	int* testInteger = falloc<int>(ctx);
	fallocDisposeCtx(ctx);
	
	// create/free stack
	fallocContext* stack = fallocCreateCtx(deviceHeap);
	fallocPush<int>(ctx, 1);
	fallocPush<int>(ctx, 2);
	int b = fallocPop<int>(ctx);
	int a = fallocPop<int>(ctx);
	fallocDisposeCtx(ctx);
}

int main()
{
	cudaPrintfInit(25600);
	cudaFallocHeap heap = cudaFallocWTraceInit(2048);

	// test
	TestFalloc<<<1, 1>>>(heap.deviceHeap);

	// trace
	fallocTrace* trace = cudaFallocTraceInit();
	void* buffer; size_t length;
	while (buffer = cudaFallocTraceStream(heap, trace, length)) {
		printf("z: %d\n", length);
	}
	cudaFallocTraceEnd(trace);

	// free and exit
	cudaFallocWTraceEnd(heap);
	cudaPrintfDisplay(stdout, true); cudaPrintfEnd();
	printf("\ndone.\n"); scanf_s("%c");
    return 0;
}
