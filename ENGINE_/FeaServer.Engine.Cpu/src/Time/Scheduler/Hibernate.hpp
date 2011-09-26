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

        __device__ struct Hibernate_t* xtor(fallocDeviceContext* falloCtx)
        {
			trace(Hibernate, "xtor");
			Elements.xtor(falloCtx);
			return this;
        }

	} Hibernate;

#endif
}}
