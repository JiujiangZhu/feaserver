//#pragma once
#include "Slice.hpp"
#include "HibernateCollection.hpp"
#include "SliceFractionCache.hpp"
#include "SliceFractionCollection.hpp"
namespace Time { namespace Scheduler {
#ifdef _SLICECOLLECTION

#elif defined(SLICECOLLECTION)
	__device__ void ElementCollection::DeHibernate(SliceCollection* slices)
    {
		ElementList::Enumerator e;
        if (_singles.GetEnumerator(e))
			for (Element* single = e.Current; !e.MoveNext(); )
            {
                ulong time = (ulong)single->Metadata;
                if (time < EngineSettings__MaxTimeslicesTime)
                    throw(Exception, "paranoia");
                ulong newTime = (ulong)(time -= EngineSettings__MaxTimeslicesTime);
                if (newTime < EngineSettings__MaxTimeslicesTime)
                {
                    _singles.Remove(single);
                    slices->Schedule(single, newTime);
                }
            }
		/*
        if (_multiples.Count > 0)
            foreach (var multiple in _multiples)
            {
                var time = BitConverter.ToUInt64(multiple.M, 0);
                if (time < EngineSettings.MaxTimeslicesTime)
                    throw new Exception("paranoia");
                var newTime = (ulong)(time -= EngineSettings.MaxTimeslicesTime);
                if (newTime < EngineSettings.MaxTimeslicesTime)
                {
                    multiples.Remove(multiple);
                    slices.Schedule(multiple.E, newTime);
                }
            }
		*/
    }

#else
	#define SLICECOLLECTION

	class SliceCollection
	{
	public:
		ulong _currentSlice;
        Slice _slices[EngineSettings__MaxTimeslices];
        HibernateCollection _hibernates;
        SliceFractionCache _fractionCache;

		__device__ SliceCollection()
			: _currentSlice(0)
		{
			trace(SliceCollection, "ctor");
			for (int sliceIndex = 0; sliceIndex < EngineSettings__MaxTimeslices; sliceIndex++)
				_slices[sliceIndex].xtor();
			_hibernates.xtor();
		}

        __device__ void Schedule(Element* element, ulong time)
        {
            trace(SliceCollection, "Schedule %d", TimePrec__DecodeTime(time));
            {
                ulong slice = (ulong)(time >> TimePrec__TimePrecisionBits);
                ulong fraction = (ulong)(time & TimePrec__TimePrecisionMask);
                if (slice < EngineSettings__MaxTimeslices)
                {
                    // first fraction
                    if (slice == 0)
                        _fractionCache.EnsureCache(fraction);
                    // roll timeslice for index
                    slice += _currentSlice;
                    if (slice >= EngineSettings__MaxTimeslices)
                        slice -= EngineSettings__MaxTimeslices;
                    _slices[slice].Fractions.Schedule(element, fraction);
                }
                else
                    _hibernates.Hibernate(element, time);
            }
        }

		/*
		__device__ void ScheduleRange(IEnumerable<Tuple<Element, ulong>> elements)
        {
            foreach (var element in elements)
                Schedule(element.Item1, element.Item2);
        }
		*/

        __device__ void MoveNextSlice()
        {
			trace(SliceCollection, "MoveNextSlice %d", _currentSlice);
			_currentSlice++;
            if (_currentSlice >= EngineSettings__MaxTimeslices)
            {
                _currentSlice = 0;
                _hibernates.DeHibernate(this);
            }
        }
	};

#endif
}}
