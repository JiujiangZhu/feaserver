#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
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
		trace(ElementCollection, "DeHibernate");
		ElementList::Enumerator e;
        if (_singles.GetEnumerator(e))
			for (Element* single = e.Current; !e.MoveNext(); )
            {
                ulong time = (ulong)single->Metadata;
                if (time < EngineSettings__MaxTimeslicesTime)
                    thrownew(Exception, "paranoia");
                ulong newTime = (ulong)(time -= EngineSettings__MaxTimeslicesTime);
                if (newTime < EngineSettings__MaxTimeslicesTime)
                {
                    _singles.Remove(single);
                    slices->Schedule(single, newTime);
                }
            }
		System::LinkedList<ElementRef>::Enumerator e2;
		if (_multiples.GetEnumerator(e2))
			for (ElementRef* multiple = e2.Current; !e2.MoveNext(); )
            {
                ulong time = (ulong)(multiple->Metadata);
                if (time < EngineSettings__MaxTimeslicesTime)
                    thrownew(Exception, "paranoia");
                ulong newTime = (ulong)(time -= EngineSettings__MaxTimeslicesTime);
                if (newTime < EngineSettings__MaxTimeslicesTime)
                {
					_multiples.Remove(multiple);
                    slices->Schedule(multiple->Element, newTime);
                }
            }
    }

#else
	#define SLICECOLLECTION

	class SliceCollection
	{
	public:
		fallocDeviceHeap* _deviceHeap;
		ulong _currentSlice;
		char _currentHibernate;
        Slice _slices[EngineSettings__MaxTimeslices];
        HibernateCollection _hibernates;
        SliceFractionCache _fractionCache;

		__device__ SliceCollection(fallocDeviceHeap* deviceHeap)
			:  _deviceHeap(deviceHeap), _currentSlice(0), _currentHibernate(0)
		{
			trace(SliceCollection, "ctor");
			for (int sliceIndex = 0; sliceIndex < EngineSettings__MaxTimeslices; sliceIndex++)
				_slices[sliceIndex].xtor(_deviceHeap);
			_hibernates.xtor(_deviceHeap);
		}
		__device__ void Dispose()
		{
			trace(SliceCollection, "Dispose");
			for (int sliceIndex = 0; sliceIndex < EngineSettings__MaxTimeslices; sliceIndex++)
				_slices[sliceIndex].Dispose();
			_hibernates.Dispose();
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
			trace(SliceCollection, "ScheduleRange");
            foreach (var element in elements)
                Schedule(element.Item1, element.Item2);
        }
		*/

        __device__ void MoveNextSlice()
        {
			trace(SliceCollection, "MoveNextSlice %d", _currentSlice);
			_slices[_currentSlice].Dispose();
            if (++_currentSlice >= EngineSettings__MaxTimeslices)
            {
                _currentSlice = 0;
                _hibernates.DeHibernate(this);
                if (++_currentHibernate >= EngineSettings__HibernatesTillReShuffle)
                {
                    _currentHibernate = 0;
                    _hibernates.ReShuffle();
                }
            }
        }
	};

#endif
}}
