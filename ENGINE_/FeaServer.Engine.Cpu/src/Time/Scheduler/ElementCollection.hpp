#pragma once
#include "ElementList.hpp"
#include "ElementRef.hpp"
namespace Time { namespace Scheduler {
#ifdef _ELEMENTCOLLECTION

#else
	#define ELEMENTCOLLECTION
	class SliceCollection;
	typedef struct ElementCollection_t
	{
	public:
		ElementList _singles;
        System::LinkedList<ElementRef> _multiples;

        __device__ struct ElementCollection_t* xtor()
        {
			trace(ElementCollection, "xtor");
			return this;
        }

        __device__ void Add(Element* element, ulong time)
        {
			trace(ElementCollection, "Add %d", TimePrec__DecodeTime(time));
            byte* metadata = (byte*)time;
			ElementRef* elementRef;
            switch (element->ScheduleStyle)
            {
				case FirstWins:
                    _singles.MergeFirstWins(element, metadata);
                    break;
               case LastWins:
                    _singles.MergeLastWins(element, metadata);
                    break;
                case Multiple:
					elementRef = nullptr;
                    _multiples.AddFirst(elementRef);
                    break;
                default:
					trace(Warn, "UNDEFINED");
                    thrownew(NotImplementedException);
            }
        }

        __device__ void Clear()
        {
			trace(ElementCollection, "Clear");
            _singles.Clear();
            _multiples.Clear();
        }

        __device__ int getCount()
        {
            return _singles.getCount() + _multiples.getCount();
        }

		/*
        __device__ IList<Element> ToList()
        {
            var list = new List<Element>();
            foreach (var singles in _singles)
                list.Add(singles);
            foreach (var multiple in _multiples)
                list.Add(multiple.E);
            return list;
        }
		*/

		__device__ void DeHibernate(SliceCollection* slices);

	} ElementCollection;

#endif
}}
