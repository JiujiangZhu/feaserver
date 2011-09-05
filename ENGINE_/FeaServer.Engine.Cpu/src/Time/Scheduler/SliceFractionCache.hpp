#pragma once
namespace Time { namespace Scheduler {
#ifdef _SLICEFRACTIONCACHE

#else
	#define SLICEFRACTIONCACHE

	class SliceFractionCache
	{
	public:
		SliceFractionCollection* _sliceFractions;
        ulong _fractions[EngineSettings__MaxWorkingFractions];
        int _currentFractionIndex;
        ulong _minFraction;
        ulong _maxFraction;
        ulong _currentFraction;
		bool RequiresRebuild;

		__device__ void EnsureCache(ulong fraction)
        {
            if (fraction < _maxFraction)
                RequiresRebuild = true;
        }
	};

#endif
}}
