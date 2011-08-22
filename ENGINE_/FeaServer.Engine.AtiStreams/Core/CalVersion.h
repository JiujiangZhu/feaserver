#pragma once
#include "cal.h"
namespace TimeServices { namespace Engine { namespace Core {
	typedef struct
	{
		CALuint major;
		CALuint minor;
		CALuint imp;

		// methods
		inline void GetVersion() { calGetVersion(&major, &minor, &imp); }
		CALint QueryVersion() { return QueryVersion(">="); }
		CALint QueryVersion(const CALchar* comparison);
	} CalVersion;
}}}