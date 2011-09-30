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

static void xmain()
{
	cpuFallocHeap heap = cpuFallocInit(1);
	fallocInit(heap.deviceHeap);

	// create/free heap
	void* obj = fallocGetChunk(heap.deviceHeap);
	fallocFreeChunk(heap.deviceHeap, obj);

	// create/free alloc
	fallocContext* ctx = fallocCreateCtx(heap.deviceHeap);
	char* testString = (char *)falloc(ctx, 10);
	int* testInteger = (int *)falloc(ctx, sizeof(int));
	fallocDisposeCtx(ctx);

	// free and exit
	cpuFallocEnd(heap);
	printf("done."); scanf_s("%c");
}
