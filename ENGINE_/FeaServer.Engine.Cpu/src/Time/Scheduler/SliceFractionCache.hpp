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

namespace Time { namespace Scheduler {
#ifdef _SLICEFRACTIONCACHE

#else
	#define SLICEFRACTIONCACHE

	class SliceFractionCache
	{
	private:
		SliceFractionCollection* _sliceFractions;
        ulong _fractions[EngineSettings__MaxWorkingFractions];
        int _currentFractionIndex;
        ulong _minFraction;
        ulong _maxFraction;
        ulong _currentFraction;

	public:
		bool RequiresRebuild;

		//__device__ SliceFraction MoveNextSliceFraction()
  //      {
  //          _minFraction = _currentFraction; //: fractions.Remove(fractionTime);
  //          _currentFraction = (_currentFractionIndex > 0 ? _fractions[--_currentFractionIndex] : ulong.MaxValue); //: repnz requires one less register
  //          return _sliceFractions[_minFraction];
  //      }

		__device__ void EnsureCache(ulong fraction)
        {
            if (fraction < _maxFraction)
                RequiresRebuild = true;
        }
	};

#endif
}}
