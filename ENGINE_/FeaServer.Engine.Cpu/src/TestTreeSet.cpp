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
using namespace System;

template class TreeSet<int>;
int TreeSet_COMPARE(unsigned __int32 shard, void* x, void* y)
{
	int a = *((int*)x);
	int b = *((int*)y);
    return (a < b ? -1 : (a > b ? 1 : 0));
}

static void amain()
{
	cpuFallocHeap heap = cpuFallocInit();
	fallocInit(heap.deviceHeap);
	fallocContext* ctx = fallocCreateCtx(heap.deviceHeap);
	fallocContext* stack = fallocCreateCtx(heap.deviceHeap);
	falloc(stack, 70, false);
	//
	TreeSet<int> treeSet; treeSet.xtor(0, ctx);
	treeSet.Add(5);
	treeSet.Add(3);
	treeSet.Add(1);
	treeSet.Add(2);
	treeSet.Add(7);
	treeSet.Add(10);

	//
	treeSet.EnumeratorBegin(stack);
	while (treeSet.EnumeratorMoveNext(stack))
		printf("%d\n", treeSet.Current);
	treeSet.EnumeratorEnd(stack);

	// free and exit
	fallocDisposeCtx(stack);
	fallocDisposeCtx(ctx);
	cpuFallocEnd(heap);
	printf("done."); scanf_s("%c");
}
