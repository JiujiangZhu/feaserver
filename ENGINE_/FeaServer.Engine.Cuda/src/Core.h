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

#define trace(type,method,...) cuPrintf(#type":"method"\n",__VA_ARGS__)
#define thrownew(type,...) { cuPrintf("\n\nTHROWS:\n"#type"\n"); __THROW; }
#include "System\cuPrintf.cu"
#include "System\cuFallocWTrace.cuh"
#include "System\LinkedList.h"
#include "System\TreeSet.h"
#include "System\SortedDictionary.h"

//typedef enum {false, true} bool;
typedef unsigned char byte;
//typedef char^std^
typedef long int decimal;
//typedef double^std^
//enum^std^
//typedef float^std^
typedef long int int_;
typedef long long int long_;
typedef signed char sbyte;
typedef short int short_;
//struct^std^
typedef unsigned long int uint;
typedef unsigned long long int ulong;
typedef unsigned short int ushort;

namespace Time {

	enum ElementScheduleStyle
	{
		FirstWins,
		LastWins,
		Multiple,
	};

	#define TimePrec__TimePrecisionBits 4
	#define TimePrec__TimePrecisionMask (ulong)4
	#define TimePrec__TimeScaler (ulong)4
	#define TimePrec__TimeScaleUnit (decimal)(1M / TimeScaler)
	#define TimePrec__EncodeTime(time) (ulong)0
	#define TimePrec__DecodeTime(time) (decimal)0

	#define EngineSettings__MaxTimeslices 10
	#define EngineSettings__MaxHibernates 1
	#define EngineSettings__MaxWorkingFractions 10
	#define EngineSettings__MaxTimeslicesTime (ulong)(EngineSettings__MaxTimeslices << TimePrec__TimePrecisionBits)
	// fixed settings
	#define EngineSettings__HibernatesTillReShuffle 3
}
