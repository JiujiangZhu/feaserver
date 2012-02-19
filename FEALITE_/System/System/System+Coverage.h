#ifndef _SYSTEM_COVERAGE_H_
#define _SYSTEM_COVERAGE_H_
#include "SystemApi+Coverage.h"
#include <assert.h>

/*
** The testcase() macro is used to aid in coverage testing.  When doing coverage testing, the condition inside the argument to
** testcase() must be evaluated both true and false in order to get full branch coverage.  The testcase() macro is inserted
** to help ensure adequate test coverage in places where simple condition/decision coverage is inadequate.  For example, testcase()
** can be used to make sure boundary values are tested.  For bitmask tests, testcase() can be used to make sure each bit
** is significant and used at least once.  On switch statements where multiple cases go to the same block of code, testcase()
** can insure that all cases are evaluated.
*/
#ifdef SYSTEM_COVERAGE_TEST
  void systemCoverage(int);
# define testcase(X)  if (X) { systemCoverage(__LINE__); }
#else
# define testcase(X)
#endif

/*
** The TESTONLY macro is used to enclose variable declarations or other bits of code that are needed to support the arguments
** within testcase() and assert() macros.
*/
#if !defined(NDEBUG) || defined(SYSTEM_COVERAGE_TEST)
# define TESTONLY(X)  X
#else
# define TESTONLY(X)
#endif

/*
** Sometimes we need a small amount of code such as a variable initialization to setup for a later assert() statement.  We do not want this code to
** appear when assert() is disabled.  The following macro is therefore used to contain that setup code.  The "VVA" acronym stands for
** "Verification, Validation, and Accreditation".  In other words, the code within VVA_ONLY() will only run during verification processes.
*/
#ifndef NDEBUG
# define VVA_ONLY(X)  X
#else
# define VVA_ONLY(X)
#endif

/*
** The ALWAYS and NEVER macros surround boolean expressions which are intended to always be true or false, respectively.  Such
** expressions could be omitted from the code completely.  But they are included in a few cases in order to enhance the resilience
** of APPID to unexpected behavior - to make the code "self-healing" or "ductile" rather than being "brittle" and crashing at the first
** hint of unplanned behavior.
** 
** In other words, ALWAYS and NEVER are added for defensive code.
** 
** When doing coverage testing ALWAYS and NEVER are hard-coded to be true and false so that the unreachable code then specify will
** not be counted as untested code.
*/
#if defined(SYSTEM_COVERAGE_TEST)
# define ALWAYS(X)      (1)
# define NEVER(X)       (0)
#elif !defined(NDEBUG)
# define ALWAYS(X)      ((X)?1:(assert(0),0))
# define NEVER(X)       ((X)?(assert(0),1):0)
#else
# define ALWAYS(X)      (X)
# define NEVER(X)       (X)
#endif

#endif /* _SYSTEM_COVERAGE_H_ */
