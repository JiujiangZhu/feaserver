#include <cuda.h>;
#include "Core.h";
#include "System\cuFalloc.cu"

__global__ void TestFalloc(fallocDeviceHeap *deviceHeap)
{
	fallocInit(deviceHeap);

	// create/free heap
	void *obj = fallocGetChunk(deviceHeap);
	fallocFreeChunk(deviceHeap, obj);

	// create/free alloc
	fallocDeviceContext *ctx = fallocCreate(deviceHeap);
	char *testString = (char *)falloc(ctx, 10);
	int *testInteger = (int *)falloc(ctx, sizeof(int));
	fallocDispose(ctx);
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
