#pragma once
#include "ElementCollection.hpp"
namespace Time { namespace Scheduler {
#ifdef _HIBERNATE

#else
	#define HIBERNATE

	typedef struct Hibernate_t
	{
	public:
        ElementCollection Elements;

        __device__ struct Hibernate_t* xtor()
        {
			trace(Hibernate, "xtor");
			Elements.xtor();
			return this;
        }

	} Hibernate;

#endif
}}
