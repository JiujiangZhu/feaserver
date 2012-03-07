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

#define __device__
#define trace(type,method,...) printf(#type":"method"\n",__VA_ARGS__)
#define thrownew(type,...) { printf("\n\nTHROWS:\n"#type); scanf_s("%c"); throw; }
#include "System\cpuFallocWTrace.h"
#include "System\LinkedList.h"
#include "System\TreeSet.h"
//#include "System\SortedDictionary.h"

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

#define array_getLength(t) *((size_t*)t-1)
#define array_getSize(T,length) ((sizeof(T)*length)+sizeof(size_t))
#define array_getSizeEx(res,length) (((res)*length)+sizeof(size_t))
#define newArray(t,T,length) (T*)((size_t*)malloc(sizeof(T)*length+4)+1);*((size_t*)t-1)=length
#define freeArray(t) free((size_t*)t-1)

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

	//#define EngineSettings__MaxTimeslices 1000
	#define EngineSettings__MaxTimeslices 10
	#define EngineSettings__MaxHibernates 1
	#define EngineSettings__MaxWorkingFractions 10
	#define EngineSettings__MaxTimeslicesTime (ulong)(EngineSettings__MaxTimeslices << TimePrec__TimePrecisionBits)
	// fixed settings
	#define EngineSettings__HibernatesTillReShuffle 3
}
