#pragma once
#include "ElementCollection.hpp"
namespace Time { namespace Scheduler {
#ifdef _SLICEFRACTION

#else
	#define SLICEFRACTION

	typedef struct SliceFraction_t
	{
	public:
		ElementCollection Elements;

        __device__ struct SliceFraction_t* xtor(fallocDeviceContext* falloCtx)
        {
			trace(SliceFraction, "xtor");
			Elements.xtor(falloCtx);
			return this;
        }

	} SliceFraction;

#endif
}}
