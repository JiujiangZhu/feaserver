#ifndef _SYSTEMAPI_CONFIG_H_
#define _SYSTEMAPI_CONFIG_H_
#ifdef __cplusplus
extern "C" {
#endif
	
/*
** API: Configuration Options
**
** These constants are the available integer configuration options that can be passed as the first argument to the [system_config()] interface.
**
** New configuration options may be added in future releases of APPID. Existing configuration options might be discontinued.  Applications
** should check the return code from [system_config()] to make sure that the call worked.  The [system_config()] interface will return a
** non-zero [error code] if a discontinued or unsupported configuration option is invoked.
**
** <dl>
** <dt>SYSTEM_CONFIG_SINGLETHREAD</dt>
** <dd>There are no arguments to this option.  ^This option sets the [threading mode] to Single-thread.  In other words, it disables
** all mutexing and puts APPID into a mode where it can only be used by a single thread.   ^If APPID is compiled with
** the [SYSTEM_THREADSAFE | SYSTEM_THREADSAFE=0] compile-time option then it is not possible to change the [threading mode] from its default
** value of Single-thread and so [system_config()] will return  [SYSTEM_ERROR] if called with the SYSTEM_CONFIG_SINGLETHREAD
** configuration option.</dd>
**
** <dt>SYSTEM_CONFIG_MULTITHREAD</dt>
** <dd>There are no arguments to this option.  ^This option sets the [threading mode] to Multi-thread.  In other words, it disables
** mutexing on [database connection] and [prepared statement] objects. The application is responsible for serializing access to
** [database connections] and [prepared statements].  But other mutexes are enabled so that APPID will be safe to use in a multi-threaded
** environment as long as no two threads attempt to use the same [database connection] at the same time.  ^If APPID is compiled with
** the [SYSTEM_THREADSAFE | SYSTEM_THREADSAFE=0] compile-time option then it is not possible to set the Multi-thread [threading mode] and
** [system_config()] will return [SYSTEM_ERROR] if called with the SYSTEM_CONFIG_MULTITHREAD configuration option.</dd>
**
** <dt>SYSTEM_CONFIG_SERIALIZED</dt>
** <dd>There are no arguments to this option.  ^This option sets the [threading mode] to Serialized. In other words, this option enables
** all mutexes including the recursive mutexes on [database connection] and [prepared statement] objects.
** In this mode (which is the default when APPID is compiled with [SYSTEM_THREADSAFE=1]) the APPID library will itself serialize access
** to [database connections] and [prepared statements] so that the application is free to use the same [database connection] or the
** same [prepared statement] in different threads at the same time. ^If APPID is compiled with
** the [SYSTEM_THREADSAFE | SYSTEM_THREADSAFE=0] compile-time option then it is not possible to set the Serialized [threading mode] and
** [system_config()] will return [SYSTEM_ERROR] if called with the SYSTEM_CONFIG_SERIALIZED configuration option.</dd>
**
** <dt>SYSTEM_CONFIG_MALLOC</dt>
** <dd> ^(This option takes a single argument which is a pointer to an instance of the [system_mem_methods] structure.  The argument specifies
** alternative low-level memory allocation routines to be used in place of the memory allocation routines built into APPID.)^ ^APPID makes
** its own private copy of the content of the [system_mem_methods] structure before the [system_config()] call returns.</dd>
**
** <dt>SYSTEM_CONFIG_GETMALLOC</dt>
** <dd> ^(This option takes a single argument which is a pointer to an instance of the [system_mem_methods] structure.  The [system_mem_methods]
** structure is filled with the currently defined memory allocation routines.)^ This option can be used to overload the default memory allocation
** routines with a wrapper that simulations memory allocation failure or tracks memory usage, for example. </dd>
**
** <dt>SYSTEM_CONFIG_MEMSTATUS</dt>
** <dd> ^This option takes single argument of type int, interpreted as a boolean, which enables or disables the collection of memory allocation 
** statistics. ^(When memory allocation statistics are disabled, the following APPID interfaces become non-operational:
**   <ul>
**   <li> [system_memory_used()]
**   <li> [system_memory_highwater()]
**   <li> [system_soft_heap_limit64()]
**   <li> [system_status()]
**   </ul>)^
** ^Memory allocation statistics are enabled by default unless APPID is compiled with [SYSTEM_DEFAULT_MEMSTATUS]=0 in which case memory
** allocation statistics are disabled by default.
** </dd>
**
** <dt>SYSTEM_CONFIG_SCRATCH</dt>
** <dd> ^This option specifies a static memory buffer that APPID can use for scratch memory.  There are three arguments:  A pointer an 8-byte
** aligned memory buffer from which the scrach allocations will be drawn, the size of each scratch allocation (sz),
** and the maximum number of scratch allocations (N).  The sz argument must be a multiple of 16.
** The first argument must be a pointer to an 8-byte aligned buffer of at least sz*N bytes of memory.
** ^APPID will use no more than two scratch buffers per thread.  So N should be set to twice the expected maximum number of threads.
** ^APPID will never require a scratch buffer that is more than 6 times the database page size. ^If APPID needs needs additional
** scratch memory beyond what is provided by this configuration option, then  [system_malloc()] will be used to obtain the memory needed.</dd>
**
** <dt>SYSTEM_CONFIG_PAGECACHE</dt>
** <dd> ^This option specifies a static memory buffer that APPID can use for the database page cache with the default page cache implemenation.  
** This configuration should not be used if an application-define page cache implementation is loaded using the SYSTEM_CONFIG_PCACHE option.
** There are three arguments to this option: A pointer to 8-byte aligned memory, the size of each page buffer (sz), and the number of pages (N).
** The sz argument should be the size of the largest database page (a power of two between 512 and 32768) plus a little extra for each
** page header.  ^The page header size is 20 to 40 bytes depending on the host architecture.  ^It is harmless, apart from the wasted memory,
** to make sz a little too large.  The first argument should point to an allocation of at least sz*N bytes of memory.
** ^APPID will use the memory provided by the first argument to satisfy its memory needs for the first N pages that it adds to cache.  ^If additional
** page cache memory is needed beyond what is provided by this option, then APPID goes to [system_malloc()] for the additional storage space.
** The pointer in the first argument must be aligned to an 8-byte boundary or subsequent behavior of APPID
** will be undefined.</dd>
**
** <dt>SYSTEM_CONFIG_HEAP</dt>
** <dd> ^This option specifies a static memory buffer that APPID will use for all of its dynamic memory allocation needs beyond those provided
** for by [SYSTEM_CONFIG_SCRATCH] and [SYSTEM_CONFIG_PAGECACHE]. There are three arguments: An 8-byte aligned pointer to the memory,
** the number of bytes in the memory buffer, and the minimum allocation size. ^If the first pointer (the memory pointer) is NULL, then APPID reverts
** to using its default memory allocator (the system malloc() implementation), undoing any prior invocation of [SYSTEM_CONFIG_MALLOC].  ^If the
** memory pointer is not NULL and either [SYSTEM_ENABLE_MEMSYS3] or [SYSTEM_ENABLE_MEMSYS5] are defined, then the alternative memory
** allocator is engaged to handle all of SQLites memory allocation needs. The first pointer (the memory pointer) must be aligned to an 8-byte
** boundary or subsequent behavior of APPID will be undefined.</dd>
**
** <dt>SYSTEM_CONFIG_MUTEX</dt>
** <dd> ^(This option takes a single argument which is a pointer to an instance of the [system_mutex_methods] structure.  The argument specifies
** alternative low-level mutex routines to be used in place the mutex routines built into APPID.)^  ^APPID makes a copy of the
** content of the [system_mutex_methods] structure before the call to [system_config()] returns. ^If APPID is compiled with
** the [SYSTEM_THREADSAFE | SYSTEM_THREADSAFE=0] compile-time option then the entire mutexing subsystem is omitted from the build and hence calls to
** [system_config()] with the SYSTEM_CONFIG_MUTEX configuration option will return [SYSTEM_ERROR].</dd>
**
** <dt>SYSTEM_CONFIG_GETMUTEX</dt>
** <dd> ^(This option takes a single argument which is a pointer to an instance of the [system_mutex_methods] structure.  The [system_mutex_methods]
** structure is filled with the currently defined mutex routines.)^ This option can be used to overload the default mutex allocation
** routines with a wrapper used to track mutex usage for performance profiling or testing, for example.   ^If APPID is compiled with
** the [SYSTEM_THREADSAFE | SYSTEM_THREADSAFE=0] compile-time option then the entire mutexing subsystem is omitted from the build and hence calls to
** [system_config()] with the SYSTEM_CONFIG_GETMUTEX configuration option will return [SYSTEM_ERROR].</dd>
**
** <dt>SYSTEM_CONFIG_LOOKASIDE</dt>
** <dd> ^(This option takes two arguments that determine the default memory allocation for the lookaside memory allocator on each
** [database connection].  The first argument is the size of each lookaside buffer slot and the second is the number of
** slots allocated to each database connection.)^  ^(This option sets the <i>default</i> lookaside size. The [SYSTEM_DBCONFIG_LOOKASIDE]
** verb to [system_db_config()] can be used to change the lookaside configuration on individual connections.)^ </dd>
**
** <dt>SYSTEM_CONFIG_PCACHE</dt>
** <dd> ^(This option takes a single argument which is a pointer to an [system_pcache_methods] object.  This object specifies the interface
** to a custom page cache implementation.)^  ^APPID makes a copy of the object and uses it for page cache memory allocations.</dd>
**
** <dt>SYSTEM_CONFIG_GETPCACHE</dt>
** <dd> ^(This option takes a single argument which is a pointer to an [system_pcache_methods] object.  APPID copies of the current
** page cache implementation into that object.)^ </dd>
**
** <dt>SYSTEM_CONFIG_LOG</dt>
** <dd> ^The SYSTEM_CONFIG_LOG option takes two arguments: a pointer to a function with a call signature of void(*)(void*,int,const char*), 
** and a pointer to void. ^If the function pointer is not NULL, it is invoked by [system_log()] to process each logging event.  ^If the
** function pointer is NULL, the [system_log()] interface becomes a no-op. ^The void pointer that is the second argument to SYSTEM_CONFIG_LOG is
** passed through as the first parameter to the application-defined logger function whenever that function is invoked.  ^The second parameter to
** the logger function is a copy of the first parameter to the corresponding [system_log()] call and is intended to be a [result code] or an
** [extended result code].  ^The third parameter passed to the logger is log message after formatting via [system_snprintf()].
** The APPID logging interface is not reentrant; the logger function supplied by the application must not invoke any APPID interface.
** In a multi-threaded application, the application-defined logger function must be threadsafe. </dd>
**
** </dl>
*/
#define SYSTEM_CONFIG_SINGLETHREAD  1  /* nil */
#define SYSTEM_CONFIG_MULTITHREAD   2  /* nil */
#define SYSTEM_CONFIG_SERIALIZED    3  /* nil */
#define SYSTEM_CONFIG_MALLOC        4  /* system_mem_methods* */
#define SYSTEM_CONFIG_GETMALLOC     5  /* system_mem_methods* */
#define SYSTEM_CONFIG_SCRATCH       6  /* void*, int sz, int N */
#define SYSTEM_CONFIG_PAGECACHE     7  /* void*, int sz, int N */
#define SYSTEM_CONFIG_HEAP          8  /* void*, int nByte, int min */
#define SYSTEM_CONFIG_MEMSTATUS     9  /* boolean */
#define SYSTEM_CONFIG_MUTEX        10  /* system_mutex_methods* */
#define SYSTEM_CONFIG_GETMUTEX     11  /* system_mutex_methods* */
/* previously SYSTEM_CONFIG_CHUNKALLOC 12 which is now unused. */ 
#define SYSTEM_CONFIG_LOOKASIDE    13  /* int int */
#define SYSTEM_CONFIG_PCACHE       14  /* system_pcache_methods* */
#define SYSTEM_CONFIG_GETPCACHE    15  /* system_pcache_methods* */
#define SYSTEM_CONFIG_LOG          16  /* xFunc, void* */


/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Configure appContext
**
** The system_ctx_config() interface is used to make configuration changes to a [app Context].  The interface is similar to
** [system_config()] except that the changes apply to a single [appContext] (specified in the first argument).  The
** system_ctx_config() interface should only be used immediately after the app context is created using [system_open()], [system_open16()], or [system_open_v2()].  
**
** The second argument to system_ctx_config(D,V,...)  is the configuration verb - an integer code that indicates what
** aspect of the [app context] is being configured. The only choice for this value is [SYSTEM_CTXCONFIG_LOOKASIDE].
** New verbs are likely to be added in future releases of APPID. Additional arguments depend on the verb.
**
** ^Calls to system_ctx_config() return SYSTEM_OK if and only if the call is considered successful.
*/
SYSTEM_API int system_ctx_config(appContext*, int op, ...);

/*
** API: Database Connection Configuration Options
**
** These constants are the available integer configuration options that can be passed as the second argument to the [system_ctx_config()] interface.
**
** New configuration options may be added in future releases of APPID. Existing configuration options might be discontinued.  Applications
** should check the return code from [system_ctx_config()] to make sure that the call worked.  ^The [system_ctx_config()] interface will return a
** non-zero [error code] if a discontinued or unsupported configuration option is invoked.
**
** <dl>
** <dt>SYSTEM_DBCONFIG_LOOKASIDE</dt>
** <dd> ^This option takes three additional arguments that determine the [lookaside memory allocator] configuration for the [app context].
** ^The first argument (the third parameter to [system_db_config()] is a pointer to an memory buffer to use for lookaside memory.
** ^The first argument after the SYSTEM_DBCONFIG_LOOKASIDE verb may be NULL in which case APPID will allocate the
** lookaside buffer itself using [system_malloc()]. ^The second argument is the size of each lookaside buffer slot.  ^The third argument is the number of
** slots.  The size of the buffer in the first argument must be greater than or equal to the product of the second and third arguments.  The buffer
** must be aligned to an 8-byte boundary.  ^If the second argument to SYSTEM_DBCONFIG_LOOKASIDE is not a multiple of 8, it is internally
** rounded down to the next smaller multiple of 8.  ^(The lookaside memory configuration for a app context can only be changed when that
** connection is not currently using lookaside memory, or in other words when the "current value" returned by [system_ctx_status](D,[SYSTEM_CONFIG_LOOKASIDE],...) is zero.
** Any attempt to change the lookaside memory configuration when lookaside memory is in use leaves the configuration unchanged and returns [SYSTEM_BUSY].)^</dd>
** </dl>
*/
#define SYSTEM_CTXCONFIG_LOOKASIDE    1001  /* void* int int */


#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_CONFIG_H_ */

