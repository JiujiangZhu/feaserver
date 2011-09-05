#pragma once
#include <stdio.h>
#include "Core.h"
#include "Time\Scheduler\SliceCollection.hpp"
using namespace Time::Scheduler;

static void main()
{
	Element e;

	//HibernateCollection h;
	//h.xtor();
	//h.Hibernate(&e, 10);
	//e.getNext(); e.getPrevious();
	

	SliceCollection* s = new SliceCollection();
	s->Schedule(&e, 10);
	s->MoveNextSlice();
	//
	printf("done."); scanf("%c");
}

#include "Time\Scheduler\SliceCollection.hpp"