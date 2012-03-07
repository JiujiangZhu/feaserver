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
#ifndef CUFALLOCWTRACE_C
#define CUFALLOCWTRACE_C
#include "cuFalloc.cu"

const static int TRACEHEAP_SIZE = 2048;

typedef struct __align__(8) _cudaFallocTrace {
	volatile __int8* trace;
	cuFallocHeapChunk* lastChunk;
	int contextIndex;
	bool complete;
	struct _cudaFallocTrace* deviceTrace;
} fallocTrace;

typedef struct {
	unsigned short magic;
	unsigned short count;
	bool free;
	bool showDetail;
} traceChunk;

// All our headers are prefixed with a magic number so we know they're ready
#define CUFALLOCTRACE_MAGIC (unsigned short)0x0A0A

__global__ void FallocWTrace(fallocDeviceHeap* deviceHeap, fallocTrace* deviceTrace) {
	volatile __int8* trace = deviceTrace->trace;
	if (!trace)
		__THROW;
	volatile __int8* endTrace = trace + TRACEHEAP_SIZE - sizeof(CUFALLOCTRACE_MAGIC);
	cuFallocHeapChunk* chunk = deviceTrace->lastChunk;
	if (!chunk) {
		chunk = (cuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap));
		// trace
		*((int*)trace) = deviceHeap->chunks; trace += sizeof(int);
	}
	volatile cuFallocHeapChunk* endChunk = (cuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap) + (CHUNKSIZEALIGN * deviceHeap->chunks));
	for (; (trace < endTrace) && (chunk < endChunk); trace += sizeof(traceChunk), chunk = (cuFallocHeapChunk*)((__int8*)chunk + (CHUNKSIZEALIGN * chunk->count))) {
		if (chunk->magic != CUFALLOC_MAGIC)
			__THROW;
		// trace
		traceChunk* w = (traceChunk*)trace;
		w->magic = CUFALLOCTRACE_MAGIC;
		w->count = chunk->count;
		if (chunk->next)
			w->free = true;
		else {
			volatile cuFallocHeapChunk* chunk2;
			for (chunk2 = deviceHeap->freeChunks; (chunk2 == chunk) || (chunk2 != nullptr); chunk2 = chunk2->next) ;
			w->free = (chunk2 == chunk);
		}
		w->showDetail = (bool)chunk->reserved;
		if ((!w->free) && (w->showDetail)) {
			/* NEED */
		}
	}
	deviceTrace->lastChunk = chunk;
	deviceTrace->complete = (trace < endTrace);
	if (deviceTrace->complete) {
		*((unsigned short*)trace) = -1; trace += sizeof(CUFALLOCTRACE_MAGIC);
	}
	deviceTrace->trace = trace;
}

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cudaFallocWTraceInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cudaFallocHeap cudaFallocWTraceInit(size_t length, cudaError_t* error) {
	cudaFallocHeap heap; memset(&heap, 0, sizeof(cudaFallocHeap));
	// Allocate a print buffer on the device and zero it
	void* deviceTrace;
	if ((!error && (cudaMalloc((void**)&deviceTrace, TRACEHEAP_SIZE) != cudaSuccess)) ||
		(error && ((*error = cudaMalloc((void**)&deviceTrace, TRACEHEAP_SIZE)) != cudaSuccess)))
		return heap;
	//
	return cudaFallocInit(length, error, deviceTrace);
}

//
//  cudaFallocWTraceEnd
//
//  Frees up the memory which we allocated
//
extern "C" void cudaFallocWTraceEnd(cudaFallocHeap &heap) {
	if (!heap.deviceHeap)
        return;
    cudaFree(heap.reserved); heap.reserved = nullptr;
	//
	cudaFallocEnd(heap);
}

//
//  cuFallocSetTraceInfo
//
//	Sets a trace Info.
//
extern "C" void cuFallocSetTraceInfo(size_t id, bool showDetail) {
}

//
//  cudaFallocTraceInit
//
//	Creates a trace Stream.
//
extern "C" fallocTrace* cudaFallocTraceInit() {
	fallocTrace* trace = (fallocTrace*)malloc(sizeof(fallocTrace)); memset(trace, 0, sizeof(fallocTrace));
	cudaMalloc(&trace->deviceTrace, sizeof(fallocTrace));
	return trace;
}

//
//  cudaFallocTraceStream
//
//	Streams till empty.
//
extern "C" void* cudaFallocTraceStream(cudaFallocHeap &heap, fallocTrace* trace, size_t &length) {
	if (trace->complete) {
		length = 0;
		return nullptr;
	}
	trace->trace = (volatile __int8*)heap.reserved;
	size_t r = cudaMemcpy(trace->deviceTrace, trace, sizeof(fallocTrace), cudaMemcpyHostToDevice);
	FallocWTrace<<<1, 1>>>(heap.deviceHeap, trace->deviceTrace);
	cudaMemcpy(trace, trace->deviceTrace, sizeof(fallocTrace), cudaMemcpyDeviceToHost);
	length = (__int8*)trace->trace - heap.reserved;
	return heap.reserved;
}

//
//  cudaFallocTraceEnd
//
//	Frees a trace Stream.
//
extern "C" void cudaFallocTraceEnd(fallocTrace* trace) {
	cudaFree(trace->deviceTrace);
	free(trace);
}

#endif // CUFALLOCWTRACE_C