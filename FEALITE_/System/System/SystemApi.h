#ifndef __SYSTEMAPI_H_
#define __SYSTEMAPI_H_
#include "SystemApi+Reset.h"
#include "SystemApi+Coverage.h"
#include "SystemApi+AppContext.h"
#include "SystemApi+Primitives.h"
#include "SystemApi+Memory.h"
#include "SystemApi+Threading.h"
#include "SystemApi+Config.h"
#include "SystemApi+Limits.h"
//
#include "SystemApi+IO.h"
#include "SystemApi+Extensions.h"
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Result Codes
** KEYWORDS: SYSTEM_OK {error code} {error codes}
** KEYWORDS: {result code} {result codes}
**
** Many APPID functions return an integer result code from the set shown here in order to indicates success or failure.
**
** New error codes may be added in future versions of APPID.
**
** See also: [SYSTEM_IOERR_READ | extended result codes]
*/
#define SYSTEM_OK           0   /* Successful result */
/* beginning-of-error-codes */
#define SYSTEM_ERROR        1   /* SQL error or missing database */
#define SYSTEM_INTERNAL     2   /* Internal logic error in APPID */
#define SYSTEM_PERM         3   /* Access permission denied */
#define SYSTEM_ABORT        4   /* Callback routine requested an abort */
#define SYSTEM_BUSY         5   /* The database file is locked */
#define SYSTEM_LOCKED       6   /* A table in the database is locked */
#define SYSTEM_NOMEM        7   /* A malloc() failed */
#define SYSTEM_READONLY     8   /* Attempt to write a readonly database */
#define SYSTEM_INTERRUPT    9   /* Operation terminated by system_interrupt()*/
#define SYSTEM_IOERR       10   /* Some kind of disk I/O error occurred */
#define SYSTEM_CORRUPT     11   /* The database disk image is malformed */
#define SYSTEM_NOTFOUND    12   /* NOT USED. Table or record not found */
#define SYSTEM_FULL        13   /* Insertion failed because database is full */
#define SYSTEM_CANTOPEN    14   /* Unable to open the database file */
#define SYSTEM_PROTOCOL    15   /* Database lock protocol error */
#define SYSTEM_EMPTY       16   /* Database is empty */
#define SYSTEM_SCHEMA      17   /* The database schema changed */
#define SYSTEM_TOOBIG      18   /* String or BLOB exceeds size limit */
#define SYSTEM_CONSTRAINT  19   /* Abort due to constraint violation */
#define SYSTEM_MISMATCH    20   /* Data type mismatch */
#define SYSTEM_MISUSE      21   /* Library used incorrectly */
#define SYSTEM_NOLFS       22   /* Uses OS features not supported on host */
#define SYSTEM_AUTH        23   /* Authorization denied */
#define SYSTEM_FORMAT      24   /* Auxiliary database format error */
#define SYSTEM_RANGE       25   /* 2nd parameter to system_bind out of range */
#define SYSTEM_NOTADB      26   /* File opened that is not a database file */
#define SYSTEM_ROW         100  /* system_step() has another row ready */
#define SYSTEM_DONE        101  /* system_step() has finished executing */
/* end-of-error-codes */

/*
** API: Error Logging Interface
**
** ^The [system_log()] interface writes a message into the error log established by the [SYSTEM_CONFIG_LOG] option to [system_config()].
** ^If logging is enabled, the zFormat string and subsequent arguments are used with [system_snprintf()] to generate the final output string.
**
** The zFormat string must not be NULL.
**
** To avoid deadlocks and other threading problems, the system_log() routine will not use dynamically allocated memory.  The log message is stored in
** a fixed-length buffer on the stack.  If the log message is longer than a few hundred characters, it will be truncated to the length of the
** buffer.
*/
SYSTEM_API void system_log(int iErrCode, const char *zFormat, ...);

/* 
** APPCONTEXT ================================================================================================================================================
*/
/*
** API: Enable Or Disable Extended Result Codes
**
** ^The system_extended_result_codes() routine enables or disables the [extended result codes] feature of APPID. ^The extended result
** codes are disabled by default for historical compatibility.
*/
SYSTEM_API int system_extended_result_codes(appContext*, int onoff);

/*
** API: Tracing And Profiling Functions
**
** These routines register callback functions that can be used for tracing and profiling the execution of SQL statements.
**
** ^The callback function registered by system_trace() is invoked at various times when an SQL statement is being run by [system_step()].
** ^The system_trace() callback is invoked with a UTF-8 rendering of the SQL statement text as the statement first begins executing.
** ^(Additional system_trace() callbacks might occur as each triggered subprogram is entered.  The callbacks for triggers
** contain a UTF-8 SQL comment that identifies the trigger.)^
**
** ^The callback function registered by system_profile() is invoked as each SQL statement finishes.  ^The profile callback contains
** the original statement text and an estimate of wall-clock time of how long that statement took to run.  ^The profile callback
** time is in units of nanoseconds, however the current implementation is only capable of millisecond resolution so the six least significant
** digits in the time are meaningless.  Future versions of APPID might provide greater resolution on the profiler callback.  The
** system_profile() function is considered experimental and is subject to change in future versions of APPID.
*/
SYSTEM_API void *system_trace(appContext*, void(*xTrace)(void*,const char*), void*);
SYSTEM_API SYSTEM_EXPERIMENTAL void *system_profile(appContext*, void(*xProfile)(void*,const char*,INT64_TYPE), void*);

/*
** API: Error Codes And Messages
**
** ^The system_errcode() interface returns the numeric [result code] or [extended result code] for the most recent failed system_* API call
** associated with a [app context]. If a prior API call failed but the most recent API call succeeded, the return value from
** system_errcode() is undefined.  ^The system_extended_errcode() interface is the same except that it always returns the 
** [extended result code] even when extended result codes are disabled.
**
** ^The system_errmsg() and system_errmsg16() return English-language text that describes the error, as either UTF-8 or UTF-16 respectively.
** ^(Memory to hold the error message string is managed internally. The application does not need to worry about freeing the result.
** However, the error string might be overwritten or deallocated by subsequent calls to other APPID interface functions.)^
**
** When the serialized [threading mode] is in use, it might be the case that a second error occurs on a separate thread in between
** the time of the first error and the call to these interfaces. When that happens, the second error will be reported since these
** interfaces always report the most recent result.  To avoid this, each thread can obtain exclusive use of the [app context] D
** by invoking [system_mutex_enter]([system_db_mutex](D)) before beginning to use D and invoking [system_mutex_leave]([system_db_mutex](D)) after
** all calls to the interfaces listed here are completed.
**
** If an interface fails with SYSTEM_MISUSE, that means the interface was invoked incorrectly by the application.  In that case, the
** error code and message may or may not be set.
*/
SYSTEM_API int system_errcode(appContext*);
SYSTEM_API int system_extended_errcode(appContext*);
SYSTEM_API const char *system_errmsg(appContext*);
SYSTEM_API const void *system_errmsg16(appContext*);

/*
** Undo the hack that converts floating point types to integer for builds on processors without floating point support.
*/
#ifdef SYSTEM_OMIT_FLOATING_POINT
# undef double
#endif
#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_H_ */

