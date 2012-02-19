#ifndef _SYSTEM_RESET_H_
#define _SYSTEM_RESET_H_
#include <stddef.h>
#include "SystemApi+Reset.h"

/*
** These #defines should enable >2GB file support on POSIX if the underlying operating system supports it.  If the OS lacks large file support, or if the OS is windows, these should be no-ops.
** The _LARGEFILE_SOURCE macro must appear before any system #includes.  Hence, this block of code must be the very first code in all source files.
** Large file support can be disabled using the -SYSTEM_DISABLE_LFS switch on the compiler command line.  This is necessary if you are compiling
** on a recent machine (ex: Red Hat 7.2) but you want your code to work on an older machine (ex: Red Hat 6.0).  If you compile on Red Hat 7.2 without this option, LFS is enable.  But LFS does not exist in the kernel
** in Red Hat 6.0, so the code won't work.  Hence, for maximum binary portability you should omit LFS.
** 
** Similar is true for Mac OS X.  LFS is only supported on Mac OS X 9 and later.
*/
#ifndef SYSTEM_DISABLE_LFS
# define _LARGE_FILE 1
# ifndef _FILE_OFFSET_BITS
#   define _FILE_OFFSET_BITS 64
# endif
# define _LARGEFILE_SOURCE 1
#endif

/*
** Disable nuisance warnings on Borland compilers
*/
#if defined(__BORLANDC__)
#pragma warn -rch // unreachable code
#pragma warn -ccc // Condition is always true or false
#pragma warn -aus // Assigned value is never used
#pragma warn -csu // Comparing signed and unsigned
#pragma warn -spa // Suspicious pointer arithmetic
#endif

/*
** Needed for various definitions...
*/
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

/*
** Many people are failing to set -DNDEBUG=1 when compiling APPID.
** Setting NDEBUG makes the code smaller and run faster.  So the following lines are added to automatically set NDEBUG unless the -DSYSTEM_DEBUG=1 option is set.
** Thus NDEBUG becomes an opt-in rather than an opt-out feature.
*/
#if !defined(NDEBUG) && !defined(SYSTEM_DEBUG) 
# define NDEBUG 1
#endif

// Include standard header files as necessary
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif
#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif

/*
** We need to define _XOPEN_SOURCE as follows in order to enable recursive mutexes on most Unix systems.  But Mac OS X is different.
** The _XOPEN_SOURCE define causes problems for Mac OS X we are told, so it is omitted there.
** 
** Later we learn that _XOPEN_SOURCE is poorly or incorrectly implemented on some systems.  So we avoid defining it at all
** if it is already defined or if it is unneeded because we are not doing a threadsafe build.
*/
#if !defined(_XOPEN_SOURCE) && !defined(__DARWIN__) && !defined(__APPLE__) && SYSTEM_THREADSAFE
#  define _XOPEN_SOURCE 500	// Needed to enable pthread recursive mutexes
#endif

/*
** The following macros are used to suppress compiler warnings and to make it clear to human readers when a function parameter is deliberately 
** left unused within the body of a function. This usually happens when a function is called via a function pointer. For example the 
** implementation of an SQL aggregate step callback may not use the parameter indicating the number of arguments passed to the aggregate,
** if it knows that this is enforced elsewhere.
** 
** When a function parameter is not used at all within the body of a function, it is generally named "NotUsed" or "NotUsed2" to make things even clearer.
** However, these macros may also be used to suppress warnings related to parameters that may or may not be used depending on compilation options.
** For example those parameters only used in assert() statements. In these cases the parameters are named as per the usual conventions.
*/
#define UNUSED_PARAMETER(x) (void)(x)
#define UNUSED_PARAMETER2(x,y) UNUSED_PARAMETER(x),UNUSED_PARAMETER(y)

/*
** GCC does not define the offsetof() macro so we'll have to do it ourselves.
*/
#ifndef offsetof
#define offsetof(STRUCTURE,FIELD) ((int)((char*)&((STRUCTURE*)0)->FIELD))
#endif

/*
** The following macros are used to cast pointers to integers and integers to pointers.  The way you do this varies from one compiler
** to the next, so we have developed the following set of #if statements to generate appropriate macros for a wide range of compilers.
** 
** The correct "ANSI" way to do this is to use the intptr_t type. 
** Unfortunately, that typedef is not available on all compilers, or if it is available, it requires an #include of specific headers
** that vary from one machine to the next.
**
** The llvm-gcc-4.2 compiler from Apple chokes on the ((void*)&((char*)0)[X]) construct.  But MSVC chokes on ((void*)(X)).
** So we have to define the macros in different ways depending on the compiler.
*/
#if defined(__PTRDIFF_TYPE__) // This case should work for GCC
# define systemInt2Ptr(X)  ((void*)(__PTRDIFF_TYPE__)(X))
# define systemPtr2Int(X)  ((int)(__PTRDIFF_TYPE__)(X))
#elif !defined(__GNUC__) // Works for compilers other than LLVM
# define systemInt2Ptr(X)  ((void*)&((char*)0)[X])
# define systemPtr2Int(X)  ((int)(((char*)X)-(char*)0))
#elif defined(HAVE_STDINT_H) // Use this case if we have ANSI headers
# define systemInt2Ptr(X)  ((void*)(intptr_t)(X))
# define systemPtr2Int(X)  ((int)(intptr_t)(X))
#else // Generates a warning - but it always works
# define systemInt2Ptr(X)  ((void*)(X))
# define systemPtr2Int(X)  ((int)(X))
#endif

/*
** When SYSTEM_OMIT_WSD is defined, it means that the target platform does not support Writable Static Data (WSD) such as global and static variables.
** All variables must either be on the stack or dynamically allocated from the heap.  When WSD is unsupported, the variable declarations scattered
** throughout the APPID code must become constants instead.  The SYSTEM_WSD macro is used for this purpose.  And instead of referencing the variable
** directly, we use its constant as a key to lookup the run-time allocated buffer that holds real variable.  The constant is also the initializer
** for the run-time allocated buffer.
** 
** In the usual case where WSD is supported, the SYSTEM_WSD and GLOBAL macros become no-ops and have zero performance impact.
*/
#ifdef SYSTEM_OMIT_WSD
  #define SYSTEM_WSD const
  #define GLOBAL(t,v) (*(t*)system_wsd_find((void*)&(v), sizeof(v)))
  int system_wsd_init(int N, int J);
  void *system_wsd_find(void *K, int L);
#else
  #define SYSTEM_WSD 
  #define GLOBAL(t,v) v
#endif

/*
** The *_BKPT macros are substitutes for the error codes with the same name but without the _BKPT suffix.  These macros invoke
** routines that report the line-number on which the error originated using system_log().  The routines also provide a convenient place
** to set a debugger breakpoint.
*/
int internalCorruptError(int);
int internalMisuseError(int);
int internalCantopenError(int);
#define SYSTEM_CORRUPT_BKPT internalCorruptError(__LINE__)
#define SYSTEM_MISUSE_BKPT internalMisuseError(__LINE__)
#define SYSTEM_CANTOPEN_BKPT internalCantopenError(__LINE__)

/*
** The ctype.h header is needed for non-ASCII systems.  It is also needed by FTS3 when FTS3 is included in the amalgamation.
*/
#if !defined(SYSTEM_ASCII) || (defined(SYSTEM_ENABLE_FTS3) && defined(SYSTEM_AMALGAMATION))
# include <ctype.h>
#endif

/*
** The macro unlikely() is a hint that surrounds a boolean expression that is usually false.  Macro likely() surrounds
** a boolean expression that is usually true.  GCC is able to use these hints to generate better code, sometimes.
*/
#if defined(__GNUC__) && 0
# define likely(X)    __builtin_expect((X),1)
# define unlikely(X)  __builtin_expect((X),0)
#else
# define likely(X)    !!(X)
# define unlikely(X)  !!(X)
#endif

/*
** INTERNAL FUNCTION PROTOTYPES
*/

#ifndef SYSTEM_AMALGAMATION
//extern SYSTEM_WSD FuncDefHash systemGlobalFunctions;
#ifndef SYSTEM_OMIT_WSD
extern int systemPendingByte;
#endif
#endif
//int systemApiExit(appContext*, int);

#endif /* _SYSTEM_RESET_H_ */
