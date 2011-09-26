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

	} HibernateCollection;

#endif
}}
