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
#ifndef CPUFALLOCWTRACE_C
#define CPUFALLOCWTRACE_C
#include "cpuFalloc.cpp"

const static int TRACEHEAP_SIZE = 2048;

typedef struct _cpuFallocTrace {
	volatile __int8* trace;
	cpuFallocHeapChunk* lastChunk;
	int contextIndex;
	bool complete;
} fallocTrace;

typedef struct {
	unsigned short magic;
	unsigned short count;
	bool free;
	bool showDetail;
} traceChunk;

// All our headers are prefixed with a magic number so we know they're ready
#define CPUFALLOCTRACE_MAGIC (unsigned short)0x0A0A

void FallocWTrace(fallocDeviceHeap* deviceHeap, fallocTrace* deviceTrace) {
	volatile __int8* trace = deviceTrace->trace;
	if (!trace)
		throw;
	volatile __int8* endTrace = trace + TRACEHEAP_SIZE - sizeof(CPUFALLOCTRACE_MAGIC);
	cpuFallocHeapChunk* chunk = deviceTrace->lastChunk;
	if (!chunk) {
		chunk = (cpuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap));
		// trace
		*((int*)trace) = (int)deviceHeap->chunks; trace += sizeof(int);
	}
	volatile cpuFallocHeapChunk* endChunk = (cpuFallocHeapChunk*)((__int8*)deviceHeap + sizeof(fallocDeviceHeap) + (CHUNKSIZEALIGN * deviceHeap->chunks));
	for (; (trace < endTrace) && (chunk < endChunk); trace += sizeof(traceChunk), chunk = (cpuFallocHeapChunk*)((__int8*)chunk + (CHUNKSIZEALIGN * chunk->count))) {
		if (chunk->magic != CPUFALLOC_MAGIC)
			throw;
		// trace
		//printf(".\n");
		traceChunk* w = (traceChunk*)trace;
		w->magic = CPUFALLOCTRACE_MAGIC;
		w->count = chunk->count;
		if (chunk->next)
			w->free = true;
		else {
			volatile cpuFallocHeapChunk* chunk2;
			for (chunk2 = deviceHeap->freeChunks; (chunk2 == chunk) || (chunk2 != nullptr); chunk2 = chunk2->next) ;
			w->free = (chunk2 == chunk);
		}
		w->showDetail = (chunk->reserved);
		if ((!w->free) && (w->showDetail)) {
			/* NEED */
		}
	}
	deviceTrace->lastChunk = chunk;
	deviceTrace->complete = (trace < endTrace);
	if (deviceTrace->complete) {
		*((unsigned short*)trace) = -1; trace += sizeof(CPUFALLOCTRACE_MAGIC);
	}
	deviceTrace->trace = trace;
}

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

//
//  cpuFallocWTraceInit
//
//  Takes a buffer length to allocate, creates the memory on the device and
//  returns a pointer to it for when a kernel is called. It's up to the caller
//  to free it.
//
extern "C" cpuFallocHeap cpuFallocWTraceInit(size_t length) {
	cpuFallocHeap heap; memset(&heap, 0, sizeof(fallocDeviceHeap));
    // allocate a print buffer on the device and zero it
	void* deviceTrace;
    if (!(deviceTrace = (void*)malloc(TRACEHEAP_SIZE)))
		return heap;
	//
	return cpuFallocInit(length, deviceTrace);
}

//
//  cpuFallocWTraceEnd
//
//  Frees up the memory which we allocated
//
extern "C" void cpuFallocWTraceEnd(cpuFallocHeap &heap) {
	if (!heap.deviceHeap)
		return;
    free(heap.reserved); heap.reserved = nullptr;
	//
	cpuFallocEnd(heap);
}

//
//  cpuFallocSetTraceInfo
//
//	Sets a trace Info.
//
extern "C" void cpuFallocSetTraceInfo(size_t id, bool showDetail) {
}

//
//  cpuFallocTraceInit
//
//	Creates a trace Stream.
//
extern "C" fallocTrace* cpuFallocTraceInit() {
	fallocTrace* trace = (fallocTrace*)malloc(sizeof(fallocTrace)); memset(trace, 0, sizeof(fallocTrace));
	return trace;
}

//
//  cpuFallocTraceStream
//
//	Streams till empty.
//
extern "C" void* cpuFallocTraceStream(cpuFallocHeap &heap, fallocTrace* trace, size_t &length) {
	if (trace->complete) {
		length = 0;
		return nullptr;
	}
	trace->trace = (volatile __int8*)heap.reserved;
	FallocWTrace(heap.deviceHeap, trace);
	length = (size_t)((__int8*)trace->trace - (__int8*)heap.reserved);
	return heap.reserved;
}

//
//  cpuFallocTraceEnd
//
//	Frees a trace Stream.
//
extern "C" void cpuFallocTraceEnd(fallocTrace* trace) {
	free(trace);
}

#endif // CPUFALLOCWTRACE_C