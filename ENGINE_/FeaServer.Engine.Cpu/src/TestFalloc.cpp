#pragma once
#include <stdio.h>
#include "Core.h"
//#include "System\cpuFalloc.cpp"

static void main()
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
	printf("done."); scanf("%c");
}
