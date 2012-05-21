/*
**
** This file contains the common header for all mutex implementations. The sqliteInt.h header #includes this file so that it is available
** to all source files.  We break it out in an effort to keep the code better organized.
**
** NOTE:  source files should *not* #include this header file directly. Source files should #include the sqliteInt.h file and let that file
** include this one indirectly.
*/

/*
** Figure out what version of the code to use.  The choices are
**
**   SYSTEM_MUTEX_OMIT         No mutex logic. Not even stubs. The mutexes implemention cannot be overridden at start-time.
**
**   SYSTEM_MUTEX_NOOP         For single-threaded applications. No mutual exclusion is provided. But this
**                             implementation can be overridden at start-time.
**
**   SYSTEM_MUTEX_UNIX     For multi-threaded applications on Unix.
**
**   SYSTEM_MUTEX_WIN          For multi-threaded applications on Win32.
*/
#if !SYSTEM_THREADSAFE
# define SYSTEM_MUTEX_OMIT
#endif
#if SYSTEM_THREADSAFE && !defined(SYSTEM_MUTEX_NOOP)
#  if SYSTEM_OS_UNIX
#    define SYSTEM_MUTEX_UNIX
#  elif SYSTEM_OS_WIN
#    define SYSTEM_MUTEX_WIN
#  else
#    define SYSTEM_MUTEX_NOOP
#  endif
#endif

#ifdef SYSTEM_MUTEX_OMIT
// If this is a no-op implementation, implement everything as macros.
#define system_mutex_alloc(X)	((system_mutex*)8)
#define system_mutex_free(X)
#define system_mutex_enter(X)
#define system_mutex_try(X)		SYSTEM_OK
#define system_mutex_leave(X)
#define system_mutex_held(X)	((void)(X),1)
#define system_mutex_notheld(X)	((void)(X),1)
#define systemMutexAlloc(X)		((system_mutex*)8)
#define systemMutexInit()		SYSTEM_OK
#define systemMutexEnd()
#endif
