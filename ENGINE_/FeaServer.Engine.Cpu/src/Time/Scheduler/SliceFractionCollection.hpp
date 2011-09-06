#pragma once
#include "SliceFraction.hpp"
namespace Time { namespace Scheduler {
#ifdef _SLICEFRACTIONCOLLECTION

#else
	#define SLICEFRACTIONCOLLECTION

	class SliceFractionCollection : public System::SortedDictionary<ulong, SliceFraction*>
	{
	public:
		__device__ SliceFractionCollection()
		{
			trace(SliceFractionCollection, "ctor");
		}

		__device__ void Schedule(Element* element, ulong fraction)
        {
			trace(SliceFractionCollection, "Schedule %d", TimePrec__DecodeTime(fraction));
            SliceFraction* fraction2;
            if (!TryGetValue(fraction, &fraction2))
                Add(fraction, fraction2 = nullptr);
            fraction2->Elements.Add(element, 0);
        }
	};

#endif
}}
