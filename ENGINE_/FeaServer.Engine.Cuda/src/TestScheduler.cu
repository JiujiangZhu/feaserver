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
#include "System\cuFalloc.cu"
#include "..\..\FeaServer.Engine.Cpu\src\Time\Element.hpp"
#include "..\..\FeaServer.Engine.Cpu\src\Time\Scheduler\SliceCollection.hpp"
using namespace Time::Scheduler;

__device__ int TreeSet_COMPARE(unsigned __int32 shard, void* x, void* y)
{
	int a = *((int*)x);
	int b = *((int*)y);
    return (a < b ? -1 : (a > b ? 1 : 0));
}

__global__ void Schedule(fallocDeviceHeap* deviceHeap)
{
	fallocInit(deviceHeap);

	Time::Element e; e.ScheduleStyle = Time::Multiple;
	e.A = 5;

	SliceCollection s; s.xtor(deviceHeap);
	s.Schedule(&e, 10);
	//s.MoveNextSlice();
	s.Dispose();
}

int main()
{
	cudaFallocHeap heap = cudaFallocInit(256000);
	cudaPrintfInit(256000);

	// schedule
	Schedule<<<1, 1>>>(heap.deviceHeap);

	// free and exit
	cudaPrintfDisplay(stdout, true); cudaPrintfEnd();
	cudaFallocEnd(heap);
	printf("\ndone.\n"); scanf_s("%c");
    return 0;
}

#include "..\..\FeaServer.Engine.Cpu\src\Time\Scheduler\SliceCollection.hpp"
