#pragma once
#include <stdio.h>
#include "Core.h"
#include "Time\Scheduler\SliceCollection.hpp"
using namespace Time::Scheduler;

static void main()
{
	cpuFallocHeap heap = cpuFallocInit();
	fallocInit(heap.deviceHeap);

	//
	Element e;
	e.ScheduleStyle = Time::Multiple;

	SliceCollection* s = new SliceCollection(heap.deviceHeap);
	s->Schedule(&e, 10);
	s->MoveNextSlice();

	// free and exit
	cpuFallocEnd(heap);
	printf("done."); scanf_s("%c");
}

#include "Time\Scheduler\SliceCollection.hpp"