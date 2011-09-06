#pragma once
#include <stdio.h>
#include "Core.h"
#include "Time\Scheduler\SliceCollection.hpp"
using namespace Time::Scheduler;

static void main()
{
	//fallocHeapInitialize(nullptr, 0);

	//
	Element e;
	e.ScheduleStyle = Time::ElementScheduleStyle::Multiple;

	SliceCollection* s = new SliceCollection();
	s->Schedule(&e, 10);
	s->MoveNextSlice();
	//
	printf("done."); scanf("%c");
}

#include "Time\Scheduler\SliceCollection.hpp"