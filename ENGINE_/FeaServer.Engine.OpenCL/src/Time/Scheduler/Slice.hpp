#pragma once
#include "SliceFractionCollection.hpp"
namespace Time { namespace Scheduler {
#ifdef _SLICE

#else
	#define SLICE

	typedef struct Slice_t
	{
	public:
		SliceFractionCollection Fractions;

        __device__ struct Slice_t* xtor()
        {
			trace(Slice, "xtor");
			return this;
        }

	} Slice;

#endif
}}
