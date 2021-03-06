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
#include <stdio.h>
#include "Core.h"

static void main()
{
	cpuFallocHeap heap = cpuFallocWTraceInit(2048); //512
	fallocInit(heap.deviceHeap);

	// create/free heap
	void* obj = fallocGetChunk(heap.deviceHeap);
	fallocFreeChunk(heap.deviceHeap, obj);
/*
	void* obj2 = fallocGetChunks(heap.deviceHeap, 144*2);
	fallocFreeChunks(heap.deviceHeap, obj2);
*/	

	// create/free alloc
	fallocContext* ctx = fallocCreateCtx(heap.deviceHeap);
	char* testString = (char*)falloc(ctx, 10);
	int* testInteger = falloc<int>(ctx);
	fallocDisposeCtx(ctx);

	// create/free stack
	fallocContext* stack = fallocCreateCtx(heap.deviceHeap);
	fallocPush<int>(ctx, 1);
	fallocPush<int>(ctx, 2);
	int b = fallocPop<int>(ctx);
	int a = fallocPop<int>(ctx);
	fallocDisposeCtx(ctx);

	// trace
	fallocTrace* trace = cpuFallocTraceInit();
	void* buffer; size_t length;
	while (buffer = cpuFallocTraceStream(heap, trace, length)) {
		printf("z: %d\n", length);
	}
	cpuFallocTraceEnd(trace);

	// free and exit
	cpuFallocWTraceEnd(heap);
	printf("done."); scanf_s("%c");
}
