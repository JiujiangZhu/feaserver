#pragma once
#include <stdio.h>
#include "Core.h"

static void xmain()
{
	cpuFallocHeap heap = cpuFallocInit(1);
	fallocInit(heap.deviceHeap);

	// create/free heap
	void *obj = fallocGetChunk(heap.deviceHeap);
	fallocFreeChunk(heap.deviceHeap, obj);

	// create/free alloc
	fallocDeviceContext *ctx = fallocCreateCtx(heap.deviceHeap);
	char *testString = (char *)falloc(ctx, 10);
	int *testInteger = (int *)falloc(ctx, sizeof(int));
	fallocDisposeCtx(ctx);

	// free and exit
	cpuFallocEnd(heap);
	printf("done."); scanf("%c");
}
