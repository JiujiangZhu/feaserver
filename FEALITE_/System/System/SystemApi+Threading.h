#ifndef _SYSTEMAPI_THREADING_H_
#define _SYSTEMAPI_THREADING_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Test To See If The Library Is Threadsafe
**
** ^The system_threadsafe() function returns zero if and only if APPID was compiled mutexing code omitted due to the
** [SYSTEM_THREADSAFE] compile-time option being set to 0.
**
** APPID can be compiled with or without mutexes.  When the [SYSTEM_THREADSAFE] C preprocessor macro is 1 or 2, mutexes
** are enabled and APPID is threadsafe.  When the [SYSTEM_THREADSAFE] macro is 0, 
** the mutexes are omitted.  Without the mutexes, it is not safe to use APPID concurrently from more than one thread.
**
** Enabling mutexes incurs a measurable performance penalty. So if speed is of utmost importance, it makes sense to disable
** the mutexes.  But for maximum safety, mutexes should be enabled. ^The default behavior is for mutexes to be enabled.
**
** This interface can be used by an application to make sure that the version of APPID that it is linking against was compiled with
** the desired setting of the [SYSTEM_THREADSAFE] macro.
**
** This interface only reports on the compile-time mutex setting of the [SYSTEM_THREADSAFE] flag.  If APPID is compiled with
** SYSTEM_THREADSAFE=1 or =2 then mutexes are enabled by default but can be fully or partially disabled using a call to [system_config()]
** with the verbs [SYSTEM_CONFIG_SINGLETHREAD], [SYSTEM_CONFIG_MULTITHREAD], or [SYSTEM_CONFIG_MUTEX].  ^(The return value of the
** system_threadsafe() function shows only the compile-time setting of thread safety, not any run-time changes to that setting made by
** system_config(). In other words, the return value from system_threadsafe() is unchanged by calls to system_config().)^
**
** See the [threading mode] documentation for additional information.
*/
SYSTEM_API int system_gThreadSafe(void);

/*
** API: Suspend Execution For A Short Time
**
** The system_sleep() function causes the current thread to suspend execution for at least a number of milliseconds specified in its parameter.
**
** If the operating system does not support sleep requests with millisecond time resolution, then the time will be rounded up to
** the nearest second. The number of milliseconds of sleep actually requested from the operating system is returned.
**
** ^APPID implements this interface by calling the xSleep() method of the default [system_vfs] object.  If the xSleep() method
** of the default VFS is not implemented correctly, or not implemented at all, then the behavior of system_sleep() may deviate from the description
** in the previous paragraphs.
*/
SYSTEM_API int system_sleep(int);

/*
** API: Mutex Handle
**
** The mutex module within APPID defines [system_mutex] to be an abstract type for a mutex object.  The APPID core never looks
** at the internal representation of an [system_mutex].  It only deals with pointers to the [system_mutex] object.
**
** Mutexes are created using [system_mutex_alloc()].
*/
typedef struct system_mutex system_mutex;

/*
** API: Mutexes
**
** The APPID core uses these routines for thread synchronization. Though they are intended for internal
** use by APPID, code that links against APPID is permitted to use any of these routines.
**
** The APPID source code contains multiple implementations of these mutex routines.  An appropriate implementation
** is selected automatically at compile-time.  ^(The following implementations are available in the APPID core:
**
** <ul>
** <li>   SYSTEM_MUTEX_OS2
** <li>   SYSTEM_MUTEX_UNIX
** <li>   SYSTEM_MUTEX_WIN
** <li>   SYSTEM_MUTEX_NOOP
** </ul>)^
**
** ^The SYSTEM_MUTEX_NOOP implementation is a set of routines that does no real locking and is appropriate for use in
** a single-threaded application.  ^The SYSTEM_MUTEX_OS2, SYSTEM_MUTEX_UNIX, and SYSTEM_MUTEX_WIN implementations
** are appropriate for use on OS/2, Unix, and Windows.
**
** ^(If APPID is compiled with the SYSTEM_MUTEX_APPDEF preprocessor macro defined (with "-DSYSTEM_MUTEX_APPDEF=1"), then no mutex
** implementation is included with the library. In this case the application must supply a custom mutex implementation using the
** [SYSTEM_CONFIG_MUTEX] option of the system_config() function before calling system_initialize() or any other public system_
** function that calls system_initialize().)^
**
** ^The system_mutex_alloc() routine allocates a new mutex and returns a pointer to it. ^If it returns NULL
** that means that a mutex could not be allocated.  ^APPID will unwind its stack and return an error.  ^(The argument
** to system_mutex_alloc() is one of these integer constants:
**
** <ul>
** <li>  SYSTEM_MUTEX_FAST
** <li>  SYSTEM_MUTEX_RECURSIVE
** <li>  SYSTEM_MUTEX_STATIC_MASTER
** <li>  SYSTEM_MUTEX_STATIC_MEM
** <li>  SYSTEM_MUTEX_STATIC_MEM2
** <li>  SYSTEM_MUTEX_STATIC_PRNG
** <li>  SYSTEM_MUTEX_STATIC_LRU
** <li>  SYSTEM_MUTEX_STATIC_LRU2
** </ul>)^
**
** ^The first two constants (SYSTEM_MUTEX_FAST and SYSTEM_MUTEX_RECURSIVE) cause system_mutex_alloc() to create
** a new mutex.  ^The new mutex is recursive when SYSTEM_MUTEX_RECURSIVE is used but not necessarily so when SYSTEM_MUTEX_FAST is used.
** The mutex implementation does not need to make a distinction between SYSTEM_MUTEX_RECURSIVE and SYSTEM_MUTEX_FAST if it does
** not want to.  ^APPID will only request a recursive mutex in cases where it really needs one.  ^If a faster non-recursive mutex
** implementation is available on the host platform, the mutex subsystem might return such a mutex in response to SYSTEM_MUTEX_FAST.
**
** ^The other allowed parameters to system_mutex_alloc() (anything other than SYSTEM_MUTEX_FAST and SYSTEM_MUTEX_RECURSIVE) each return
** a pointer to a static preexisting mutex.  ^Six static mutexes are used by the current version of APPID.  Future versions of APPID
** may add additional static mutexes.  Static mutexes are for internal use by APPID only.  Applications that use APPID mutexes should
** use only the dynamic mutexes returned by SYSTEM_MUTEX_FAST or SYSTEM_MUTEX_RECURSIVE.
**
** ^Note that if one of the dynamic mutex parameters (SYSTEME_MUTEX_FAST or SYSTEM_MUTEX_RECURSIVE) is used then system_mutex_alloc()
** returns a different mutex on every call.  ^But for the static mutex types, the same mutex is returned on every call that has
** the same type number.
**
** ^The system_mutex_free() routine deallocates a previously allocated dynamic mutex.  ^APPID is careful to deallocate every
** dynamic mutex that it allocates.  The dynamic mutexes must not be in use when they are deallocated.  Attempting to deallocate a static
** mutex results in undefined behavior.  ^APPID never deallocates a static mutex.
**
** ^The system_mutex_enter() and system_mutex_try() routines attempt to enter a mutex.  ^If another thread is already within the mutex,
** system_mutex_enter() will block and system_mutex_try() will return SYSTEM_BUSY.  ^The system_mutex_try() interface returns [SYSTEM_OK]
** upon successful entry.  ^(Mutexes created using SYSTEM_MUTEX_RECURSIVE can be entered multiple times by the same thread.
** In such cases the, mutex must be exited an equal number of times before another thread can enter.)^  ^(If the same thread tries to enter any other
** kind of mutex more than once, the behavior is undefined. APPID will never exhibit such behavior in its own use of mutexes.)^
**
** ^(Some systems (for example, Windows 95) do not support the operation implemented by system_mutex_try().  On those systems, system_mutex_try()
** will always return SYSTEM_BUSY.  The APPID core only ever uses system_mutex_try() as an optimization so this is acceptable behavior.)^
**
** ^The system_mutex_leave() routine exits a mutex that was previously entered by the same thread.   ^(The behavior is undefined
** if the mutex is not currently entered by the calling thread or is not currently allocated.  APPID will never do either.)^
**
** ^If the argument to system_mutex_enter(), system_mutex_try(), or system_mutex_leave() is a NULL pointer, then all three routines behave as no-ops.
**
** See also: [system_mutex_held()] and [system_mutex_notheld()].
*/
SYSTEM_API system_mutex *system_mutex_alloc(int);
SYSTEM_API void system_mutex_free(system_mutex*);
SYSTEM_API void system_mutex_enter(system_mutex*);
SYSTEM_API int system_mutex_try(system_mutex*);
SYSTEM_API void system_mutex_leave(system_mutex*);

/*
** API: Mutex Methods Object
**
** An instance of this structure defines the low-level routines used to allocate and use mutexes.
**
** Usually, the default mutex implementations provided by APPID are sufficient, however the user has the option of substituting a custom
** implementation for specialized deployments or systems for which APPID does not provide a suitable implementation. In this case, the user
** creates and populates an instance of this structure to pass to system_config() along with the [SYSTEM_CONFIG_MUTEX] option.
** Additionally, an instance of this structure can be used as an output variable when querying the system for the current mutex
** implementation, using the [SYSTEM_CONFIG_GETMUTEX] option.
**
** ^The xMutexInit method defined by this structure is invoked as part of system initialization by the system_initialize() function.
** ^The xMutexInit routine is called by APPID exactly once for each effective call to [system_initialize()].
**
** ^The xMutexEnd method defined by this structure is invoked as part of system shutdown by the system_shutdown() function. The
** implementation of this method is expected to release all outstanding resources obtained by the mutex methods implementation, especially
** those obtained by the xMutexInit method.  ^The xMutexEnd() interface is invoked exactly once for each call to [system_shutdown()].
**
** ^(The remaining seven methods defined by this structure (xMutexAlloc, xMutexFree, xMutexEnter, xMutexTry, xMutexLeave, xMutexHeld and
** xMutexNotheld) implement the following interfaces (respectively):
**
** <ul>
**   <li>  [system_mutex_alloc()] </li>
**   <li>  [system_mutex_free()] </li>
**   <li>  [system_mutex_enter()] </li>
**   <li>  [system_mutex_try()] </li>
**   <li>  [system_mutex_leave()] </li>
**   <li>  [system_mutex_held()] </li>
**   <li>  [system_mutex_notheld()] </li>
** </ul>)^
**
** The only difference is that the public system_XXX functions enumerated above silently ignore any invocations that pass a NULL pointer instead
** of a valid mutex handle. The implementations of the methods defined by this structure are not required to handle this case, the results
** of passing a NULL pointer instead of a valid mutex handle are undefined (i.e. it is acceptable to provide an implementation that segfaults if
** it is passed a NULL pointer).
**
** The xMutexInit() method must be threadsafe.  ^It must be harmless to invoke xMutexInit() multiple times within the same process and without
** intervening calls to xMutexEnd().  Second and subsequent calls to xMutexInit() must be no-ops.
**
** ^xMutexInit() must not use APPID memory allocation ([system_malloc()] and its associates).  ^Similarly, xMutexAlloc() must not use APPID memory
** allocation for a static mutex.  ^However xMutexAlloc() may use APPID memory allocation for a fast or recursive mutex.
**
** ^APPID will invoke the xMutexEnd() method when [system_shutdown()] is called, but only if the prior call to xMutexInit returned SYSTEM_OK.
** If xMutexInit fails in any way, it is expected to clean up after itself prior to returning.
*/
typedef struct system_mutex_methods system_mutex_methods;
struct system_mutex_methods
{
	int (*xMutexInit)(void);
	int (*xMutexEnd)(void);
	system_mutex *(*xMutexAlloc)(int);
	void (*xMutexFree)(system_mutex*);
	void (*xMutexEnter)(system_mutex*);
	int (*xMutexTry)(system_mutex*);
	void (*xMutexLeave)(system_mutex*);
	int (*xMutexHeld)(system_mutex*);
	int (*xMutexNotheld)(system_mutex*);
};

/*
** API: Mutex Verification Routines
**
** The system_mutex_held() and system_mutex_notheld() routines are intended for use inside assert() statements.  ^The APPID core
** never uses these routines except inside an assert() and applications are advised to follow the lead of the core.  ^The APPID core only
** provides implementations for these routines when it is compiled with the SYSTEM_DEBUG flag.  ^External mutex implementations
** are only required to provide these routines if SYSTEM_DEBUG is defined and if NDEBUG is not defined.
**
** ^These routines should return true if the mutex in their argument is held or not held, respectively, by the calling thread.
**
** ^The implementation is not required to provided versions of these routines that actually work. If the implementation does not provide working
** versions of these routines, it should at least provide stubs that always return true so that one does not get spurious assertion failures.
**
** ^If the argument to system_mutex_held() is a NULL pointer then the routine should return 1.   This seems counter-intuitive since
** clearly the mutex cannot be held if it does not exist.  But the the reason the mutex does not exist is because the build is not
** using mutexes.  And we do not want the assert() containing the call to system_mutex_held() to fail, so a non-zero return is
** the appropriate thing to do.  ^The system_mutex_notheld() interface should also return 1 when given a NULL pointer.
*/
#ifndef NDEBUG
SYSTEM_API int system_mutex_held(system_mutex*);
SYSTEM_API int system_mutex_notheld(system_mutex*);
#endif

/*
** API: Mutex Types
**
** The [system_mutex_alloc()] interface takes a single argument which is one of these integer constants.
**
** The set of static mutexes may change from one APPID release to the next.  Applications that override the built-in mutex logic must be
** prepared to accommodate additional static mutexes.
*/
#define SYSTEM_MUTEX_FAST             0
#define SYSTEM_MUTEX_RECURSIVE        1
#define SYSTEM_MUTEX_STATIC_MASTER    2
#define SYSTEM_MUTEX_STATIC_MEM       3  /* system_malloc() */
#define SYSTEM_MUTEX_STATIC_MEM2      4  /* NOT USED */
#define SYSTEM_MUTEX_STATIC_OPEN      4  /* systemBtreeOpen() */
#define SYSTEM_MUTEX_STATIC_PRNG      5  /* system_random() */
#define SYSTEM_MUTEX_STATIC_LRU       6  /* lru page list */
#define SYSTEM_MUTEX_STATIC_LRU2      7  /* lru page list */


/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Retrieve the mutex for a app context
**
** ^This interface returns a pointer the [system_mutex] object that serializes access to the [app context] given in the argument
** when the [threading mode] is Serialized. ^If the [threading mode] is Single-thread or Multi-thread then this
** routine returns a NULL pointer.
*/
SYSTEM_API system_mutex *system_ctx_mutex(appContext*);

/*
** API: Interrupt A Long-Running Query
**
** ^This function causes any pending database operation to abort and return at its earliest opportunity. This routine is typically
** called in response to a user action such as pressing "Cancel" or Ctrl-C where the user wants a long query operation to halt
** immediately.
**
** ^It is safe to call this routine from a thread different from the thread that is currently running the database operation.  But it
** is not safe to call this routine with a [app context] that is closed or might close before system_interrupt() returns.
**
** ^If an SQL operation is very nearly finished at the time when system_interrupt() is called, then it might not have an opportunity
** to be interrupted and might continue to completion.
**
** ^An SQL operation that is interrupted will return [SYSTEM_INTERRUPT]. ^If the interrupted SQL operation is an INSERT, UPDATE, or DELETE
** that is inside an explicit transaction, then the entire transaction will be rolled back automatically.
**
** ^The system_interrupt(D) call is in effect until all currently running SQL statements on [app context] D complete.  ^Any new SQL statements
** that are started after the system_interrupt() call and before the running statements reaches zero are interrupted as if they had been
** running prior to the system_interrupt() call.  ^New SQL statements that are started after the running statement count reaches zero are
** not effected by the system_interrupt(). ^A call to system_interrupt(D) that occurs when there are no running
** SQL statements is a no-op and has no effect on SQL statements that are started after the system_interrupt() call returns.
**
** If the app context closes while [system_interrupt()] is running then bad things will likely happen.
*/
SYSTEM_API void system_interrupt(appContext*);

/*
** API: Register A Callback To Handle SYSTEM_BUSY Errors
**
** ^This routine sets a callback function that might be invoked whenever an attempt is made to open a database table that another thread
** or process has locked.
**
** ^If the busy callback is NULL, then [SYSTEM_BUSY] or [SYSTEM_IOERR_BLOCKED] is returned immediately upon encountering the lock.  ^If the busy callback
** is not NULL, then the callback might be invoked with two arguments.
**
** ^The first argument to the busy handler is a copy of the void* pointer which is the third argument to system_busy_handler().  ^The second argument to
** the busy handler callback is the number of times that the busy handler has been invoked for this locking event.  ^If the
** busy callback returns 0, then no additional attempts are made to access the database and [SYSTEM_BUSY] or [SYSTEM_IOERR_BLOCKED] is returned.
** ^If the callback returns non-zero, then another attempt is made to open the database for reading and the cycle repeats.
**
** The presence of a busy handler does not guarantee that it will be invoked when there is lock contention. ^If APPID determines that invoking the busy
** handler could result in a deadlock, it will go ahead and return [SYSTEM_BUSY] or [SYSTEM_IOERR_BLOCKED] instead of invoking the busy handler.
** Consider a scenario where one process is holding a read lock that it is trying to promote to a reserved lock and
** a second process is holding a reserved lock that it is trying to promote to an exclusive lock.  The first process cannot proceed
** because it is blocked by the second and the second process cannot proceed because it is blocked by the first.  If both processes
** invoke the busy handlers, neither will make any progress.  Therefore, APPID returns [SYSTEM_BUSY] for the first process, hoping that this
** will induce the first process to release its read lock and allow the second process to proceed.
**
** ^The default busy callback is NULL.
**
** ^The [SYSTEM_BUSY] error is converted to [SYSTEM_IOERR_BLOCKED] when APPID is in the middle of a large transaction where all the
** changes will not fit into the in-memory cache.  APPID will already hold a RESERVED lock on the database file, but it needs
** to promote this lock to EXCLUSIVE so that it can spill cache pages into the database file without harm to concurrent
** readers.  ^If it is unable to promote the lock, then the in-memory cache will be left in an inconsistent state and so the error
** code is promoted from the relatively benign [SYSTEM_BUSY] to the more severe [SYSTEM_IOERR_BLOCKED].  ^This error code promotion
** forces an automatic rollback of the changes.  See the <a href="/cvstrac/wiki?p=CorruptionFollowingBusyError">
** CorruptionFollowingBusyError</a> wiki page for a discussion of why this is important.
**
** ^(There can only be a single busy handler defined for each [app context].  Setting a new busy handler clears any
** previously set handler.)^  ^Note that calling [system_busy_timeout()] will also set or clear the busy handler.
**
** The busy callback should not take any actions which modify the app context that invoked the busy handler.  Any such actions result in undefined behavior.
** 
** A busy handler must not close the app context or [prepared statement] that invoked the busy handler.
*/
SYSTEM_API int system_busy_handler(appContext*, int(*)(void*,int), void*);

/*
** API: Set A Busy Timeout
**
** ^This routine sets a [system_busy_handler | busy handler] that sleeps for a specified amount of time when a table is locked.  ^The handler
** will sleep multiple times until at least "ms" milliseconds of sleeping have accumulated.  ^After at least "ms" milliseconds of sleeping,
** the handler returns 0 which causes [system_step()] to return [SYSTEM_BUSY] or [SYSTEM_IOERR_BLOCKED].
**
** ^Calling this routine with an argument less than or equal to zero turns off all busy handlers.
**
** ^(There can only be a single busy handler for a particular [app context] any any given moment.  If another busy handler
** was defined  (using [system_busy_handler()]) prior to calling this routine, that other busy handler is cleared.)^
*/
SYSTEM_API int system_busy_timeout(appContext*, int ms);

/*
** API: Query Progress Callbacks
**
** ^The system_progress_handler(D,N,X,P) interface causes the callback function X to be invoked periodically during long running calls to
** [system_exec()], [system_step()] and [system_get_table()] for app context D.  An example use for this
** interface is to keep a GUI updated during a large query.
**
** ^The parameter P is passed through as the only parameter to the  callback function X.  ^The parameter N is the number of 
** [virtual machine instructions] that are evaluated between successive invocations of the callback X.
**
** ^Only a single progress handler may be defined at one time per [app context]; setting a new progress handler cancels the
** old one.  ^Setting parameter X to NULL disables the progress handler. ^The progress handler is also disabled by setting N to a value less
** than 1.
**
** ^If the progress callback returns non-zero, the operation is interrupted.  This feature can be used to implement a
** "Cancel" button on a GUI progress dialog box.
**
** The progress handler callback must not do anything that will modify the app context that invoked the progress handler.
** Note that [system_prepare_v2()] and [system_step()] both modify their app contexts for the meaning of "modify" in this paragraph.
**
*/
SYSTEM_API void system_progress_handler(appContext*, int, int(*)(void*), void*);

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_THREADING_H_ */
