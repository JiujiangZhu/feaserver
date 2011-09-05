#pragma once
#include "SliceNode.hpp"
namespace Time { namespace Scheduler {
#ifdef _SLICEFRACTIONCOLLECTION

#else
	#define SLICEFRACTIONCOLLECTION

	class SliceFractionCollection
	{
	public:
		__device__ SliceFractionCollection()
		{
			trace(SliceFractionCollection, "ctor");
		}

		__device__ void Schedule(Element* element, ulong fraction)
        {
            SliceNode node;
            //if (!TryGetValue(fraction, out node))
            //    Add(fraction, node = new SliceNode(0));
            node.Elements.Add(element, 0);
        }
	};

#endif
}}
