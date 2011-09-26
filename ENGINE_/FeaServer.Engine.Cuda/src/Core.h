#pragma once

#define trace(type,method,...) cuPrintf(#type":"method"\n",__VA_ARGS__)
#define thrownew(type,...) { cuPrintf("\n\nTHROWS:\n"#type); __THROW; }
#include "System\cuPrintf.cu"
#include "System\cuFalloc.cuh"
#include "System\LinkedList.h"
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
}
