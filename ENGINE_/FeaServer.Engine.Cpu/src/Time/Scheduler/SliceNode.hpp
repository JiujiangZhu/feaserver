#pragma once
#include "ElementCollection.hpp"
namespace Time { namespace Scheduler {
#ifdef _SLICENODE

#else
	#define SLICENODE

	typedef struct SliceNode_t
	{
	public:
		ElementCollection Elements;

        __device__ struct SliceNode_t* xtor()
        {
			trace(SliceNode, "xtor");
			Elements.xtor();
			return this;
        }

	} SliceNode;

#endif
}}
