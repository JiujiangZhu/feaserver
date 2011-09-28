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
#include <memory.h>
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
		fallocDeviceContext* _falloCtx;
		ElementList _singles;
        System::LinkedList<ElementRef> _multiples;

        __device__ struct ElementCollection_t* xtor(fallocDeviceContext* falloCtx)
        {
			trace(ElementCollection, "xtor");
			_falloCtx = falloCtx;
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
					elementRef = (ElementRef*)falloc(_falloCtx, sizeof(ElementRef));
					if (elementRef == nullptr)
						thrownew(OutOfMemoryException);
					elementRef->Element = element;
					if (metadata != nullptr)
						memcpy(elementRef->Metadata, metadata, MetadataSize);
					//else
					//	memset(elementRef->Metadata, 0, MetadataSize);
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
