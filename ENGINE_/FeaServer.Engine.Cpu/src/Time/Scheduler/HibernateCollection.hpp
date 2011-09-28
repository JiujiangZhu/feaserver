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
#pragma once
#include "Hibernate.hpp"

namespace Time { namespace Scheduler {
#ifdef _HIBERNATECOLLECTION

#else
	#define HIBERNATECOLLECTION
	class SliceCollection;
	typedef struct HibernateCollection_t
	{
	public:
		fallocDeviceContext* _falloCtx;
		Hibernate _hibernates[EngineSettings__MaxHibernates];

        __device__ struct HibernateCollection_t* xtor(fallocDeviceHeap* deviceHeap)
        {
			trace(HibernateCollection, "xtor");
			_falloCtx = fallocCreateCtx(deviceHeap);
            for (int hibernateIndex = 0; hibernateIndex < EngineSettings__MaxHibernates; hibernateIndex++)
                _hibernates[hibernateIndex].xtor(_falloCtx);
			return this;
        }
		__device__ void Dispose()
		{
			trace(HibernateCollection, "Dispose");
			fallocDisposeCtx(_falloCtx);
		}

        __device__ void Hibernate(Element* element, ulong time)
        {
			trace(HibernateCollection, "Hibernate %d", TimePrec__DecodeTime(time));
            Scheduler::Hibernate* hibernate = &_hibernates[0];
            hibernate->Elements.Add(element, time);
        }

		 __device__ void DeHibernate(SliceCollection* slices)
        {
			trace(HibernateCollection, "DeHibernate");
            Scheduler::Hibernate* hibernate = &_hibernates[0];
            hibernate->Elements.DeHibernate(slices);
        }

        __device__ void ReShuffle()
        {
			trace(HibernateCollection, "ReShuffle");
        }

	} HibernateCollection;

#endif
}}
