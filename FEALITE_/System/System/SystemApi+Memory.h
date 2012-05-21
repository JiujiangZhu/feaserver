#ifndef _SYSTEMAPI_MEMORY_H_
#define _SYSTEMAPI_MEMORY_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Constants Defining Special Destructor Behavior
**
** These are special values for the destructor that is passed in as the final argument to routines like [system_result_blob()].  ^If the destructor
** argument is SYSTEM_STATIC, it means that the content pointer is constant and will never change.  It does not need to be destroyed.  ^The
** SYSTEM_TRANSIENT value means that the content will likely change in the near future and that APPID should make its own private copy of
** the content before returning.
**
** The typedef is necessary to work around problems in certain C++ compilers.
*/
typedef void (*system_destructor_type)(void*);
#define SYSTEM_STATIC      ((system_destructor_type)0)
#define SYSTEM_TRANSIENT   ((system_destructor_type)-1)

/*
** API: Memory Allocation Routines
**
** An instance of this object defines the interface between APPID and low-level memory allocation routines.
**
** This object is used in only one place in the APPID interface. A pointer to an instance of this object is the argument to
** [system_config()] when the configuration option is [SYSTEM_CONFIG_MALLOC] or [SYSTEM_CONFIG_GETMALLOC].  
** By creating an instance of this object and passing it to [system_config]([SYSTEM_CONFIG_MALLOC])
** during configuration, an application can specify an alternative memory allocation subsystem for APPID to use for all of its
** dynamic memory needs.
**
** Note that APPID comes with several [built-in memory allocators] that are perfectly adequate for the overwhelming majority of applications
** and that this object is only useful to a tiny minority of applications with specialized memory allocation requirements.  This object is
** also used during testing of APPID in order to specify an alternative memory allocator that simulates memory out-of-memory conditions in
** order to verify that APPID recovers gracefully from such conditions.
**
** The xMalloc and xFree methods must work like the malloc() and free() functions from the standard C library.
** The xRealloc method must work like realloc() from the standard C library with the exception that if the second argument to xRealloc is zero,
** xRealloc must be a no-op - it must not perform any allocation or deallocation.  ^APPID guarantees that the second argument to
** xRealloc is always a value returned by a prior call to xRoundup. And so in cases where xRoundup always returns a positive number,
** xRealloc can perform exactly as the standard library realloc() and still be in compliance with this specification.
**
** xSize should return the allocated size of a memory allocation previously obtained from xMalloc or xRealloc.  The allocated size
** is always at least as big as the requested size but may be larger.
**
** The xRoundup method returns what would be the allocated size of a memory allocation given a particular requested size.  Most memory
** allocators round up memory allocations at least to the next multiple of 8.  Some allocators round up to a larger multiple or to a power of 2.
** Every memory allocation request coming in through [system_malloc()] or [system_realloc()] first calls xRoundup.  If xRoundup returns 0, 
** that causes the corresponding memory allocation to fail.
**
** The xInit method initializes the memory allocator.  (For example, it might allocate any require mutexes or initialize internal data
** structures.  The xShutdown method is invoked (indirectly) by [system_shutdown()] and should deallocate any resources acquired
** by xInit.  The pAppData pointer is used as the only parameter to xInit and xShutdown.
**
** APPID holds the [SYSTEM_MUTEX_STATIC_MASTER] mutex when it invokes the xInit method, so the xInit method need not be threadsafe.  The
** xShutdown method is only called from [system_shutdown()] so it does not need to be threadsafe either.  For all other methods, APPID
** holds the [SYSTEM_MUTEX_STATIC_MEM] mutex as long as the [SYSTEM_CONFIG_MEMSTATUS] configuration option is turned on (which
** it is by default) and so the methods are automatically serialized. However, if [SYSTEM_CONFIG_MEMSTATUS] is disabled, then the other
** methods must be threadsafe or else make their own arrangements for serialization.
**
** APPID will never invoke xInit() more than once without an intervening call to xShutdown().
*/
typedef struct system_mem_methods system_mem_methods;
struct system_mem_methods
{
	void *(*xMalloc)(int);         /* Memory allocation function */
	void (*xFree)(void*);          /* Free a prior allocation */
	void *(*xRealloc)(void*,int);  /* Resize an allocation */
	int (*xSize)(void*);           /* Return the size of an allocation */
	int (*xRoundup)(int);          /* Round up request size to allocation size */
	int (*xInit)(void*);           /* Initialize the memory allocator */
	void (*xShutdown)(void*);      /* Deinitialize the memory allocator */
	void *pAppData;                /* Argument to xInit() and xShutdown() */
};

/*
** API: Memory Allocation Subsystem
**
** The APPID core uses these three routines for all of its own internal memory allocation needs. "Core" in the previous sentence
** does not include operating-system specific VFS implementation.  The Windows VFS uses native malloc() and free() for some operations.
**
** ^The system_malloc() routine returns a pointer to a block of memory at least N bytes in length, where N is the parameter.
** ^If system_malloc() is unable to obtain sufficient free memory, it returns a NULL pointer.  ^If the parameter N to
** system_malloc() is zero or negative then system_malloc() returns a NULL pointer.
**
** ^Calling system_free() with a pointer previously returned by system_malloc() or system_realloc() releases that memory so
** that it might be reused.  ^The system_free() routine is a no-op if is called with a NULL pointer.  Passing a NULL pointer
** to system_free() is harmless.  After being freed, memory should neither be read nor written.  Even reading previously freed
** memory might result in a segmentation fault or other severe error. Memory corruption, a segmentation fault, or other severe error
** might result if system_free() is called with a non-NULL pointer that was not obtained from system_malloc() or system_realloc().
**
** ^(The system_realloc() interface attempts to resize a prior memory allocation to be at least N bytes, where N is the
** second parameter.  The memory allocation to be resized is the first parameter.)^ ^ If the first parameter to system_realloc()
** is a NULL pointer then its behavior is identical to calling system_malloc(N) where N is the second parameter to system_realloc().
** ^If the second parameter to system_realloc() is zero or negative then the behavior is exactly the same as calling
** system_free(P) where P is the first parameter to system_realloc(). ^system_realloc() returns a pointer to a memory allocation
** of at least N bytes in size or NULL if sufficient memory is unavailable. ^If M is the size of the prior allocation, then min(N,M) bytes
** of the prior allocation are copied into the beginning of buffer returned by system_realloc() and the prior allocation is freed.
** ^If system_realloc() returns NULL, then the prior allocation is not freed.
**
** ^The memory returned by system_malloc() and system_realloc() is always aligned to at least an 8 byte boundary, or to a
** 4 byte boundary if the [SYSTEM_4_BYTE_ALIGNED_MALLOC] compile-time option is used.
**
** The Windows OS interface layer calls the system malloc() and free() directly when converting
** filenames between the UTF-8 encoding used by APPID and whatever filename encoding is used by the particular Windows
** installation.  Memory allocation errors are detected, but they are reported back as [SYSTEM_CANTOPEN] or
** [SYSTEM_IOERR] rather than [SYSTEM_NOMEM].
**
** The pointer arguments to [system_free()] and [system_realloc()] must be either NULL or else pointers obtained from a prior
** invocation of [system_malloc()] or [system_realloc()] that have not yet been released.
**
** The application must not read or write any part of a block of memory after it has been released using
** [system_free()] or [system_realloc()].
*/
SYSTEM_API void *system_malloc(int);
SYSTEM_API void *system_realloc(void*, int);
SYSTEM_API void system_free(void*);

/*
** API: Memory Allocator Statistics
**
** APPID provides these two interfaces for reporting on the status of the [system_malloc()], [system_free()], and [system_realloc()]
** routines, which form the built-in memory allocation subsystem.
**
** ^The [system_memory_used()] routine returns the number of bytes of memory currently outstanding (malloced but not freed).
** ^The [system_memory_highwater()] routine returns the maximum value of [system_memory_used()] since the high-water mark
** was last reset.  ^The values returned by [system_memory_used()] and [system_memory_highwater()] include any overhead
** added by APPID in its implementation of [system_malloc()], but not overhead added by the any underlying system library
** routines that [system_malloc()] may call.
**
** ^The memory high-water mark is reset to the current value of [system_memory_used()] if and only if the parameter to
** [system_memory_highwater()] is true.  ^The value returned by [system_memory_highwater(1)] is the high-water mark
** prior to the reset.
*/
SYSTEM_API INT64_TYPE system_memory_used(void);
SYSTEM_API INT64_TYPE system_memory_highwater(int resetFlag);

/*
** API: Attempt To Free Heap Memory
**
** ^The system_release_memory() interface attempts to free N bytes of heap memory by deallocating non-essential memory allocations
** held by the APPID library.   Memory used to cache database pages to improve performance is an example of non-essential memory.
** ^system_release_memory() returns the number of bytes actually freed, which might be more or less than the amount requested.
** ^The system_release_memory() routine is a no-op returning zero if APPID is not compiled with [SYSTEM_ENABLE_MEMORY_MANAGEMENT].
*/
SYSTEM_API int system_release_memory(int);

/*
** API: Impose A Limit On Heap Size
**
** ^The system_soft_heap_limit64() interface sets and/or queries the soft limit on the amount of heap memory that may be allocated by APPID.
** ^APPID strives to keep heap memory utilization below the soft heap limit by reducing the number of pages held in the page cache
** as heap memory usages approaches the limit. ^The soft heap limit is "soft" because even though APPID strives to stay
** below the limit, it will exceed the limit rather than generate an [SYSTEM_NOMEM] error.  In other words, the soft heap limit 
** is advisory only.
**
** ^The return value from system_soft_heap_limit64() is the size of the soft heap limit prior to the call.  ^If the argument N is negative
** then no change is made to the soft heap limit.  Hence, the current size of the soft heap limit can be determined by invoking
** system_soft_heap_limit64() with a negative argument.
**
** ^If the argument N is zero then the soft heap limit is disabled.
**
** ^(The soft heap limit is not enforced in the current implementation if one or more of following conditions are true:
**
** <ul>
** <li> The soft heap limit is set to zero.
** <li> Memory accounting is disabled using a combination of the [system_config]([SYSTEM_CONFIG_MEMSTATUS],...) start-time option and
**      the [SYSTEM_DEFAULT_MEMSTATUS] compile-time option.
** <li> An alternative page cache implementation is specifed using [system_config]([SYSTEM_CONFIG_PCACHE],...).
** <li> The page cache allocates from its own memory pool supplied by [system_config]([SYSTEM_CONFIG_PAGECACHE],...) rather than
**      from the heap.
** </ul>)^
**
** The soft heap limit is enforced regardless of whether or not the [SYSTEM_ENABLE_MEMORY_MANAGEMENT]
** compile-time option is invoked.  With [SYSTEM_ENABLE_MEMORY_MANAGEMENT], the soft heap limit is enforced on every memory allocation.  Without
** [SYSTEM_ENABLE_MEMORY_MANAGEMENT], the soft heap limit is only enforced when memory is allocated by the page cache.  Testing suggests that because
** the page cache is the predominate memory user in APPID, most applications will achieve adequate soft heap limit enforcement without
** the use of [SYSTEM_ENABLE_MEMORY_MANAGEMENT].
**
** The circumstances under which APPID will enforce the soft heap limit may changes in future releases of APPID.
*/
SYSTEM_API INT64_TYPE system_soft_heap_limit64(INT64_TYPE N);

/*
** API: APPID Runtime Status
**
** ^This interface is used to retrieve runtime status information about the performance of APPID, and optionally to reset various
** highwater marks.  ^The first argument is an integer code for the specific parameter to measure.  ^(Recognized integer codes
** are of the form [SYSTEM_STATUS_MEMORY_USED | SYSTEM_STATUS_...].)^ ^The current value of the parameter is returned into *pCurrent.
** ^The highest recorded value is returned in *pHighwater.  ^If the resetFlag is true, then the highest record value is reset after
** *pHighwater is written.  ^(Some parameters do not record the highest value.  For those parameters
** nothing is written into *pHighwater and the resetFlag is ignored.)^ ^(Other parameters record only the highwater mark and not the current
** value.  For these latter parameters nothing is written into *pCurrent.)^
**
** ^The system_memstatus() routine returns SYSTEM_OK on success and a non-zero [error code] on failure.
**
** This routine is threadsafe but is not atomic.  This routine can be called while other threads are running the same or different APPID
** interfaces.  However the values returned in *pCurrent and *pHighwater reflect the status of APPID at different points in time
** and it is possible that another thread might change the parameter in between the times when *pCurrent and *pHighwater are written.
**
** See also: [system_ctx_memstatus()]
*/
SYSTEM_API int system_memstatus(int op, int *pCurrent, int *pHighwater, int resetFlag);

/*
** API: Status Parameters
**
** These integer constants designate various run-time status parameters that can be returned by [system_memstatus()].
**
** <dl>
** ^(<dt>SYSTEM_STATUS_MEMORY_USED</dt>
** <dd>This parameter is the current amount of memory checked out using [system_malloc()], either directly or indirectly.  The
** figure includes calls made to [system_malloc()] by the application and internal memory usage by the APPID library.  Scratch memory
** controlled by [SYSTEM_CONFIG_SCRATCH] and auxiliary page-cache memory controlled by [SYSTEM_CONFIG_PAGECACHE] is not included in
** this parameter.  The amount returned is the sum of the allocation sizes as reported by the xSize method in [system_mem_methods].</dd>)^
**
** ^(<dt>SYSTEM_STATUS_MALLOC_SIZE</dt>
** <dd>This parameter records the largest memory allocation request handed to [system_malloc()] or [system_realloc()] (or their
** internal equivalents).  Only the value returned in the *pHighwater parameter to [system_memstatus()] is of interest.  
** The value written into the *pCurrent parameter is undefined.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_MALLOC_COUNT</dt>
** <dd>This parameter records the number of separate memory allocations.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_PAGECACHE_USED</dt>
** <dd>This parameter returns the number of pages used out of the [pagecache memory allocator] that was configured using 
** [SYSTEM_CONFIG_PAGECACHE].  The value returned is in pages, not in bytes.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_PAGECACHE_OVERFLOW</dt>
** <dd>This parameter returns the number of bytes of page cache allocation which could not be satisfied by the [SYSTEM_CONFIG_PAGECACHE]
** buffer and where forced to overflow to [system_malloc()].  The returned value includes allocations that overflowed because they
** where too large (they were larger than the "sz" parameter to [SYSTEM_CONFIG_PAGECACHE]) and allocations that overflowed because
** no space was left in the page cache.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_PAGECACHE_SIZE</dt>
** <dd>This parameter records the largest memory allocation request handed to [pagecache memory allocator].  Only the value returned in the
** *pHighwater parameter to [system_memstatus()] is of interest.  The value written into the *pCurrent parameter is undefined.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_SCRATCH_USED</dt>
** <dd>This parameter returns the number of allocations used out of the [scratch memory allocator] configured using
** [SYSTEM_CONFIG_SCRATCH].  The value returned is in allocations, not in bytes.  Since a single thread may only have one scratch allocation
** outstanding at time, this parameter also reports the number of threads using scratch memory at the same time.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_SCRATCH_OVERFLOW</dt>
** <dd>This parameter returns the number of bytes of scratch memory allocation which could not be satisfied by the [SYSTEM_CONFIG_SCRATCH]
** buffer and where forced to overflow to [system_malloc()].  The values returned include overflows because the requested allocation was too
** larger (that is, because the requested allocation was larger than the "sz" parameter to [SYSTEM_CONFIG_SCRATCH]) and because no scratch buffer
** slots were available.
** </dd>)^
**
** ^(<dt>SYSTEM_STATUS_SCRATCH_SIZE</dt>
** <dd>This parameter records the largest memory allocation request handed to [scratch memory allocator].  Only the value returned in the
** *pHighwater parameter to [system_memstatus()] is of interest.  The value written into the *pCurrent parameter is undefined.</dd>)^
**
** ^(<dt>SYSTEM_STATUS_PARSER_STACK</dt>
** <dd>This parameter records the deepest parser stack.  It is only meaningful if APPID is compiled with [YYTRACKMAXSTACKDEPTH].</dd>)^
** </dl>
**
** New status parameters may be added from time to time.
*/
#define SYSTEM_MEMSTATUS_MEMORY_USED          0
#define SYSTEM_MEMSTATUS_PAGECACHE_USED       1
#define SYSTEM_MEMSTATUS_PAGECACHE_OVERFLOW   2
#define SYSTEM_MEMSTATUS_SCRATCH_USED         3
#define SYSTEM_MEMSTATUS_SCRATCH_OVERFLOW     4
#define SYSTEM_MEMSTATUS_MALLOC_SIZE          5
#define SYSTEM_MEMSTATUS_PARSER_STACK         6
#define SYSTEM_MEMSTATUS_PAGECACHE_SIZE       7
#define SYSTEM_MEMSTATUS_SCRATCH_SIZE         8
#define SYSTEM_MEMSTATUS_MALLOC_COUNT         9

/*
** API: Custom Page Cache Object
**
** The system_pcache type is opaque.  It is implemented by the pluggable module.  The APPID core has no knowledge of
** its size or internal structure and never deals with the system_pcache object except by holding and passing pointers
** to the object.
**
** See [system_pcache_methods] for additional information.
*/
typedef struct system_pcache system_pcache;

/*
** API: Application Defined Page Cache.
** KEYWORDS: {page cache}
**
** ^(The [system_config]([SYSTEM_CONFIG_PCACHE], ...) interface can register an alternative page cache implementation by passing in an 
** instance of the system_pcache_methods structure.)^ In many applications, most of the heap memory allocated by APPID is used for the page cache.
** By implementing a custom page cache using this API, an application can better control
** the amount of memory consumed by APPID, the way in which that memory is allocated and released, and the policies used to 
** determine exactly which parts of a database file are cached and for how long.
**
** The alternative page cache mechanism is an extreme measure that is only needed by the most demanding applications.
** The built-in page cache is recommended for most uses.
**
** ^(The contents of the system_pcache_methods structure are copied to an internal buffer by APPID within the call to [system_config].  Hence
** the application may discard the parameter after the call to [system_config()] returns.)^
**
** ^(The xInit() method is called once for each effective  call to [system_initialize()])^
** (usually only once during the lifetime of the process). ^(The xInit() method is passed a copy of the system_pcache_methods.pArg value.)^
** The intent of the xInit() method is to set up global data structures required by the custom page cache implementation. 
** ^(If the xInit() method is NULL, then the built-in default page cache is used instead of the application defined
** page cache.)^
**
** ^The xShutdown() method is called by [system_shutdown()]. It can be used to clean up 
** any outstanding resources before process shutdown, if required. ^The xShutdown() method may be NULL.
**
** ^APPID automatically serializes calls to the xInit method, so the xInit method need not be threadsafe.  ^The
** xShutdown method is only called from [system_shutdown()] so it does not need to be threadsafe either.  All other methods must be threadsafe
** in multithreaded applications.
**
** ^APPID will never invoke xInit() more than once without an intervening call to xShutdown().
**
** ^APPID invokes the xCreate() method to construct a new cache instance. APPID will typically create one cache instance for each open database file,
** though this is not guaranteed. ^The first parameter, szPage, is the size in bytes of the pages that must
** be allocated by the cache.  ^szPage will not be a power of two.  ^szPage will the page size of the database file that is to be cached plus an
** increment (here called "R") of about 100 or 200.  APPID will use the extra R bytes on each page to store metadata about the underlying
** database page on disk.  The value of R depends on the APPID version, the target platform, and how APPID was compiled.
** ^R is constant for a particular build of APPID.  ^The second argument to xCreate(), bPurgeable, is true if the cache being created will
** be used to cache database pages of a file stored on disk, or false if it is used for an in-memory database. The cache implementation
** does not have to do anything special based with the value of bPurgeable; it is purely advisory.  ^On a cache where bPurgeable is false, APPID will
** never invoke xUnpin() except to deliberately delete a page. ^In other words, calls to xUnpin() on a cache with bPurgeable set to
** false will always have the "discard" flag set to true.  ^Hence, a cache created with bPurgeable false will never contain any unpinned pages.
**
** ^(The xCachesize() method may be called at any time by APPID to set the suggested maximum cache-size (number of pages stored by) the cache
** instance passed as the first argument. This is the value configured using the APPID "[PRAGMA cache_size]" command.)^  As with the bPurgeable
** parameter, the implementation is not required to do anything with this value; it is advisory only.
**
** The xPagecount() method must return the number of pages currently stored in the cache, both pinned and unpinned.
** 
** The xFetch() method locates a page in the cache and returns a pointer to the page, or a NULL pointer.
** A "page", in this context, means a buffer of szPage bytes aligned at an 8-byte boundary. The page to be fetched is determined by the key. ^The
** mimimum key value is 1.  After it has been retrieved using xFetch, the page is considered to be "pinned".
**
** If the requested page is already in the page cache, then the page cache implementation must return a pointer to the page buffer with its content
** intact.  If the requested page is not already in the cache, then the behavior of the cache implementation should use the value of the createFlag
** parameter to help it determined what action to take:
**
** <table border=1 width=85% align=center>
** <tr><th> createFlag <th> Behaviour when page is not already in cache
** <tr><td> 0 <td> Do not allocate a new page.  Return NULL.
** <tr><td> 1 <td> Allocate a new page if it easy and convenient to do so. Otherwise return NULL.
** <tr><td> 2 <td> Make every effort to allocate a new page.  Only return NULL if allocating a new page is effectively impossible.
** </table>
**
** ^(APPID will normally invoke xFetch() with a createFlag of 0 or 1.  APPID will only use a createFlag of 2 after a prior call with a createFlag of 1
** failed.)^  In between the to xFetch() calls, APPID may attempt to unpin one or more cache pages by spilling the content of
** pinned pages to disk and synching the operating system disk cache.
**
** ^xUnpin() is called by APPID with a pointer to a currently pinned page as its second argument.  If the third parameter, discard, is non-zero,
** then the page must be evicted from the cache. ^If the discard parameter is zero, then the page may be discarded or retained at the discretion of
** page cache implementation. ^The page cache implementation may choose to evict unpinned pages at any time.
**
** The cache must not perform any reference counting. A single call to xUnpin() unpins the page regardless of the number of prior calls 
** to xFetch().
**
** The xRekey() method is used to change the key value associated with the page passed as the second argument. If the cache
** previously contains an entry associated with newKey, it must be discarded. ^Any prior cache entry associated with newKey is guaranteed not
** to be pinned.
**
** When APPID calls the xTruncate() method, the cache must discard all existing cache entries with page numbers (keys) greater than or equal
** to the value of the iLimit parameter passed to xTruncate(). If any of these pages are pinned, they are implicitly unpinned, meaning that
** they can be safely discarded.
**
** ^The xDestroy() method is used to delete a cache allocated by xCreate(). All resources associated with the specified cache should be freed. ^After
** calling the xDestroy() method, APPID considers the [system_pcache*] handle invalid, and will not use it with any other system_pcache_methods
** functions.
*/
typedef struct system_pcache_methods system_pcache_methods;
struct system_pcache_methods
{
	void *pArg;
	int (*xInit)(void*);
	void (*xShutdown)(void*);
	system_pcache *(*xCreate)(int szPage, int bPurgeable);
	void (*xCachesize)(system_pcache*, int nCachesize);
	int (*xPagecount)(system_pcache*);
	void *(*xFetch)(system_pcache*, unsigned key, int createFlag);
	void (*xUnpin)(system_pcache*, void*, int discard);
	void (*xRekey)(system_pcache*, void*, unsigned oldKey, unsigned newKey);
	void (*xTruncate)(system_pcache*, unsigned iLimit);
	void (*xDestroy)(system_pcache*);
};


/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Database Connection Status
**
** ^This interface is used to retrieve runtime status information about a single [app context].  ^The first argument is the
** app context object to be interrogated.  ^The second argument is an integer constant, taken from the set of
** [SYSTEM_DBSTATUS_LOOKASIDE_USED | SYSTEM_DBSTATUS_*] macros, that determines the parameter to interrogate.  The set of 
** [SYSTEM_DBSTATUS_LOOKASIDE_USED | SYSTEM_DBSTATUS_*] macros is likely to grow in future releases of APPID.
**
** ^The current value of the requested parameter is written into *pCur and the highest instantaneous value is written into *pHiwtr.  ^If
** the resetFlg is true, then the highest instantaneous value is reset back down to the current value.
**
** ^The system_db_status() routine returns SYSTEM_OK on success and a non-zero [error code] on failure.
**
** See also: [system_memstatus()] and [system_stmt_status()].
*/
SYSTEM_API int system_ctx_memstatus(appContext*, int op, int *pCur, int *pHiwtr, int resetFlg);

/*
** API: Status Parameters for app contexts
**
** These constants are the available integer "verbs" that can be passed as the second argument to the [system_db_status()] interface.
**
** New verbs may be added in future releases of APPID. Existing verbs might be discontinued. Applications should check the return code from
** [system_db_status()] to make sure that the call worked. The [system_db_status()] interface will return a non-zero error code
** if a discontinued or unsupported verb is invoked.
**
** <dl>
** ^(<dt>SYSTEM_DBSTATUS_LOOKASIDE_USED</dt>
** <dd>This parameter returns the number of lookaside memory slots currently checked out.</dd>)^
**
** ^(<dt>SYSTEM_DBSTATUS_CACHE_USED</dt>
** <dd>This parameter returns the approximate number of of bytes of heap memory used by all pager caches associated with the app context.)^
** ^The highwater mark associated with SYSTEM_DBSTATUS_CACHE_USED is always 0.
**
** ^(<dt>SYSTEM_DBSTATUS_SCHEMA_USED</dt>
** <dd>This parameter returns the approximate number of of bytes of heap memory used to store the schema for all databases associated
** with the connection - main, temp, and any [ATTACH]-ed databases.)^  ^The full amount of memory used by the schemas is reported, even if the
** schema memory is shared with other app contexts due to [shared cache mode] being enabled.
** ^The highwater mark associated with SYSTEM_DBSTATUS_SCHEMA_USED is always 0.
**
** ^(<dt>SYSTEM_DBSTATUS_STMT_USED</dt>
** <dd>This parameter returns the approximate number of of bytes of heap and lookaside memory used by all prepared statements associated with
** the app context.)^ ^The highwater mark associated with SYSTEM_DBSTATUS_STMT_USED is always 0.
** </dd>
** </dl>
*/
#define SYSTEM_CTXMEMSTATUS_LOOKASIDE_USED     0
#define SYSTEM_CTXMEMSTATUS_CACHE_USED         1
#define SYSTEM_CTXMEMSTATUS_SCHEMA_USED        2
#define SYSTEM_CTXMEMSTATUS_STMT_USED          3
#define SYSTEM_CTXMEMSTATUS_MAX                3   /* Largest defined DBSTATUS */

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_MEMORY_H_ */

