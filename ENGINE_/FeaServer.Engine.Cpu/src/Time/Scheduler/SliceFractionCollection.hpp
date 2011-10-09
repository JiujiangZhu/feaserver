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
#include "SliceFraction.hpp"

namespace Time { namespace Scheduler {
#ifdef _SLICEFRACTIONCOLLECTION

#else
	#define SLICEFRACTIONCOLLECTION

	typedef struct { ulong key; SliceFraction* value; } SliceFractionPair;
	class SliceFractionCollection
	{
	private:
		System::TreeSet<SliceFractionPair> _set;
		fallocContext* _fallocCtx;

		__device__ bool TryGetValue(ulong key, SliceFraction** value)
		{
			SliceFractionPair pair;
			pair.key = key;
			System::Node<SliceFractionPair>* node = _set.FindNode(pair);
			if (node == nullptr)
				return false;
			*value = node->item.value;
			return true;
		}

		__device__ void Add(ulong key, SliceFraction* value)
		{
			SliceFractionPair pair;
			pair.key = key; pair.value = value;
			_set.Add(pair);
		}

	public:
		__device__ void xtor(fallocContext* fallocCtx)
		{
			trace(SliceFractionCollection, "xtor");
			_fallocCtx = fallocCtx;
			_set.xtor(0, fallocCtx);
		}

		__device__ void Schedule(Element* element, ulong fraction)
        {
			trace(SliceFractionCollection, "Schedule %d", TimePrec__DecodeTime(fraction));
            SliceFraction* fraction2;
            if (!TryGetValue(fraction, &fraction2))
			{
				fraction2 = falloc<SliceFraction>(_fallocCtx);
				fraction2->xtor(_fallocCtx);
                Add(fraction, fraction2);
			}
            fraction2->Elements.Add(element, 0);
        }

	};

#endif
}}
