/*
** This file contains macros and a little bit of code that is common to all of the platform-specific files (os_*.c) and is #included into those files.
**
** This file should be #included by the os_*.c files only.  It is not a general purpose header file.
*/
#ifndef _SYSTEM_OS_COMMON_H_
#define _SYSTEM_OS_COMMON_H_

#ifdef SYSTEM_DEBUG
int systemOSTrace = 0;
#define OSTRACE(X)		if (systemOSTrace) systemDebugPrintf X
#else
#define OSTRACE(X)
#endif

/*
** Macros for performance tracing.  Normally turned off.  Only works on i486 hardware.
*/
#ifdef SYSTEM_PERFORMANCE_TRACE

/* 
** hwtime.h contains inline assembler code for implementing high-performance timing routines.
*/
#include "hwtime.h"

static INT64_TYPE g_start;
static INT64_TYPE g_elapsed;
#define TIMER_START       g_start=sqlite3Hwtime()
#define TIMER_END         g_elapsed=sqlite3Hwtime()-g_start
#define TIMER_ELAPSED     g_elapsed
#else
#define TIMER_START
#define TIMER_END
#define TIMER_ELAPSED     ((INT64_TYPE)0)
#endif

/*
** If we compile with the SQLITE_TEST macro set, then the following block of code will give us the ability to simulate a disk I/O error.  This
** is used for testing the I/O recovery logic.
*/
#ifdef SYSTEM_TEST
int system_io_error_hit = 0;            /* Total number of I/O Errors */
int system_io_error_hardhit = 0;        /* Number of non-benign errors */
int system_io_error_pending = 0;        /* Count down to first I/O error */
int system_io_error_persist = 0;        /* True if I/O errors persist */
int system_io_error_benign = 0;         /* True if errors are benign */
int system_diskfull_pending = 0;
int system_diskfull = 0;
#define SimulateIOErrorBenign(X) system3_io_error_benign=(X)
#define SimulateIOError(CODE)  \
	if ((system_io_error_persist && sqlite3_io_error_hit) || sqlite3_io_error_pending-- == 1)  { local_ioerr(); CODE; }
static void local_ioerr()
{
	IOTRACE(("IOERR\n"));
	system_io_error_hit++;
	if (!system_io_error_benign)
		system_io_error_hardhit++;
}
#define SimulateDiskfullError(CODE) \
	if (system_diskfull_pending) { \
		if (system_diskfull_pending == 1) { \
			local_ioerr(); \
			system_diskfull = 1; \
			system_io_error_hit = 1; \
			CODE; \
		} else { \
			system_diskfull_pending--; \
		} \
	}
#else
#define SimulateIOErrorBenign(X)
#define SimulateIOError(A)
#define SimulateDiskfullError(A)
#endif

/*
** When testing, keep a count of the number of open files.
*/
#ifdef SQLITE_TEST
int system_open_file_count = 0;
#define OpenCounter(X)  system_open_file_count += (X)
#else
#define OpenCounter(X)
#endif

#endif /* _SYSTEM_OS_COMMON_H_ */
