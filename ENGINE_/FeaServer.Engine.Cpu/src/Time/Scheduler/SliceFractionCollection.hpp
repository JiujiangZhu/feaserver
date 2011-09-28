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

	class SliceFractionCollection : public System::SortedDictionary<ulong, SliceFraction*>
	{
	public:
		fallocDeviceContext* _falloCtx;

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
			{
				fraction2 = (SliceFraction*)falloc(_falloCtx, sizeof(SliceFraction));
				fraction2->xtor(_falloCtx);
                Add(fraction, fraction2);
			}
            fraction2->Elements.Add(element, 0);
        }
	};

#endif
}}
