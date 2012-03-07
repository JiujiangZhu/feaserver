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
#ifndef CPUFALLOCWSTREAM_H
#define CPUFALLOCWSTREAM_H
#include "cpuFalloc.h"

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code

typedef struct _cpuFallocTrace fallocTrace;

//
//	cpuFallocWTraceInit
//
//	Call this to initialise a falloc heap. If the buffer size needs to be changed, call cudaFallocEnd()
//	before re-calling cudaFallocInit().
//
//	The default size for the buffer is 1 megabyte. For CUDA
//	architecture 1.1 and above, the buffer is filled linearly and
//	is completely used.
//
//	Arguments:
//		length - Length, in bytes, of total space to reserve (in device global memory) for output.
//
//	Returns:
//		cudaSuccess if all is well.
//
extern "C" cpuFallocHeap cpuFallocWTraceInit(size_t length=1048576);   // 1-meg

//
//	cpuFallocWTraceEnd
//
//	Cleans up all memories allocated by cudaFallocInit() for a heap.
//	Call this at exit, or before calling cudaFallocInit() again.
//
extern "C" void cpuFallocWTraceEnd(cpuFallocHeap &heap);

//
//  cpuFallocSetTraceInfo
//
//	Sets a trace Info.
//
extern "C" void cpuFallocSetTraceInfo(size_t id, bool showDetail);

//
//  cpuFallocTraceInit
//
//	Creates a trace Stream.
//
extern "C" fallocTrace* cpuFallocTraceInit();

//
//	cpuFallocTrace
//
//	Streams till empty.
//
extern "C" void* cpuFallocTraceStream(cpuFallocHeap &heap, fallocTrace* trace, size_t &length);

//
//  cpuFallocTraceEnd
//
//	Frees a trace Stream.
//
extern "C" void cpuFallocTraceEnd(fallocTrace* trace);

#endif // CPUFALLOCWSTREAM_H
