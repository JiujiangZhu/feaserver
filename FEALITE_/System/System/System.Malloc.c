/*
** Memory allocation functions used throughout sqlite.
*/
#include "System.h"
#include <stdarg.h>

/*
** Attempt to release up to n bytes of non-essential memory currently held by APPID. An example of non-essential memory is memory used to
** cache database pages that are not currently in use.
*/
int system_release_memory(int n)
{
#ifdef SYSTEM_ENABLE_MEMORY_MANAGEMENT
	return systemPcacheReleaseMemory(n);
#else
	/* IMPLEMENTATION-OF: R-34391-24921 The system_release_memory() routine is a no-op returning zero if APPID is not compiled with
	** SYSTEM_ENABLE_MEMORY_MANAGEMENT. */
	UNUSED_PARAMETER(n);
	return 0;
#endif
}

/*
** An instance of the following object records the location of each unused scratch buffer.
*/
typedef struct ScratchFreeslot
{
	struct ScratchFreeslot *pNext;   /* Next unused scratch buffer */
} ScratchFreeslot;

/*
** State information local to the memory allocation subsystem.
*/
static SYSTEM_WSD struct Mem0Global
{
	system_mutex *mutex;         /* Mutex to serialize access */
	/*
	** The alarm callback and its arguments.  The mem0.mutex lock will be held while the callback is running.  Recursive calls into
	** the memory subsystem are allowed, but no new callbacks will be issued.
	*/
	i64 alarmThreshold;
	void (*alarmCallback)(void*, i64,int);
	void *alarmArg;
	/*
	** Pointers to the end of systemGlobalConfig.pScratch memory (so that a range test can be used to determine if an allocation
	** being freed came from pScratch) and a pointer to the list of unused scratch allocations.
	*/
	void *pScratchEnd;
	ScratchFreeslot *pScratchFree;
	u32 nScratchFree;
	/*
	** True if heap is nearly "full" where "full" is defined by the system_soft_heap_limit() setting.
	*/
	int nearlyFull;
} mem0 = { 0, 0, 0, 0, 0, 0, 0, 0 };

#define mem0 GLOBAL(struct Mem0Global, mem0)

/*
** This routine runs when the memory allocator sees that the total memory allocation is about to exceed the soft heap
** limit.
*/
static void softHeapLimitEnforcer(void *NotUsed, i64 NotUsed2, int allocSize)
{
	UNUSED_PARAMETER2(NotUsed, NotUsed2);
	system_release_memory(allocSize);
}

/*
** Change the alarm callback
*/
static int systemMemoryAlarm(void(*xCallback)(void *pArg, i64 used,int N), void *pArg, i64 iThreshold)
{
	int nUsed;
	system_mutex_enter(mem0.mutex);
	mem0.alarmCallback = xCallback;
	mem0.alarmArg = pArg;
	mem0.alarmThreshold = iThreshold;
	nUsed = systemStatusValue(SYSTEM_MEMSTATUS_MEMORY_USED);
	mem0.nearlyFull = (iThreshold > 0 && iThreshold <= nUsed);
	system_mutex_leave(mem0.mutex);
	return SYSTEM_OK;
}

/*
** Set the soft heap-size limit for the library. Passing a zero or negative value indicates no limit.
*/
i64 system_soft_heap_limit64(i64 n)
{
	i64 priorLimit;
	i64 excess;
#ifndef SYSTEM_OMIT_AUTOINIT
	system_initialize();
#endif
	system_mutex_enter(mem0.mutex);
	priorLimit = mem0.alarmThreshold;
	system_mutex_leave(mem0.mutex);
	if (n < 0)
		return priorLimit;
	if (n > 0)
		systemMemoryAlarm(softHeapLimitEnforcer, 0, n);
	else
		systemMemoryAlarm(0, 0, 0);
	excess = system_memory_used() - n;
	if (excess > 0)
		system_release_memory((int)(excess & 0x7fffffff));
	return priorLimit;
}

void system_soft_heap_limit(int n)
{
	if (n < 0)
		n = 0;
	system_soft_heap_limit64(n);
}

/*
** Initialize the memory allocation subsystem.
*/
int systemMallocInit(void)
{
	if (systemGlobalConfig.m.xMalloc == 0)
		systemMemSetDefault();
	memset(&mem0, 0, sizeof(mem0));
	if (systemGlobalConfig.bCoreMutex)
		mem0.mutex = systemMutexAlloc(SYSTEM_MUTEX_STATIC_MEM);
	if (systemGlobalConfig.pScratch && systemGlobalConfig.szScratch >= 100 && systemGlobalConfig.nScratch > 0)
	{
		int i, n, sz;
		ScratchFreeslot *pSlot;
		sz = ROUNDDOWN8(systemGlobalConfig.szScratch);
		systemGlobalConfig.szScratch = sz;
		pSlot = (ScratchFreeslot*)systemGlobalConfig.pScratch;
		n = systemGlobalConfig.nScratch;
		mem0.pScratchFree = pSlot;
		mem0.nScratchFree = n;
		for (i = 0; i < n-1; i++)
		{
			pSlot->pNext = (ScratchFreeslot*)(sz+(char*)pSlot);
			pSlot = pSlot->pNext;
		}
		pSlot->pNext = 0;
		mem0.pScratchEnd = (void*)&pSlot[1];
	}
	else
	{
		mem0.pScratchEnd = 0;
		systemGlobalConfig.pScratch = 0;
		systemGlobalConfig.szScratch = 0;
		systemGlobalConfig.nScratch = 0;
	}
	if (systemGlobalConfig.pPage == 0 || systemGlobalConfig.szPage < 512 || systemGlobalConfig.nPage < 1)
	{
		systemGlobalConfig.pPage = 0;
		systemGlobalConfig.szPage = 0;
		systemGlobalConfig.nPage = 0;
	}
	return systemGlobalConfig.m.xInit(systemGlobalConfig.m.pAppData);
}

/*
** Return true if the heap is currently under memory pressure - in other words if the amount of heap used is close to the limit set by
** system_soft_heap_limit().
*/
int systemHeapNearlyFull(void)
{
	return mem0.nearlyFull;
}

/*
** Deinitialize the memory allocation subsystem.
*/
void systemMallocEnd(void)
{
	if (systemGlobalConfig.m.xShutdown)
		systemGlobalConfig.m.xShutdown(systemGlobalConfig.m.pAppData);
	memset(&mem0, 0, sizeof(mem0));
}

/*
** Return the amount of memory currently checked out.
*/
i64 system_memory_used(void)
{
	int n, mx;
	i64 res;
	system_memstatus(SYSTEM_MEMSTATUS_MEMORY_USED, &n, &mx, 0);
	res = (i64)n;  /* Work around bug in Borland C. Ticket #3216 */
	return res;
}

/*
** Return the maximum amount of memory that has ever been checked out since either the beginning of this process
** or since the most recent reset.
*/
i64 system_memory_highwater(int resetFlag)
{
	int n, mx;
	i64 res;
	system_memstatus(SYSTEM_MEMSTATUS_MEMORY_USED, &n, &mx, resetFlag);
	res = (i64)mx;  /* Work around bug in Borland C. Ticket #3216 */
	return res;
}

/*
** Trigger the alarm 
*/
static void systemMallocAlarm(int nByte)
{
	void (*xCallback)(void*,i64,int);
	i64 nowUsed;
	void *pArg;
	if( mem0.alarmCallback==0 ) return;
	xCallback = mem0.alarmCallback;
	nowUsed = systemStatusValue(SYSTEM_MEMSTATUS_MEMORY_USED);
	pArg = mem0.alarmArg;
	mem0.alarmCallback = 0;
	system_mutex_leave(mem0.mutex);
	xCallback(pArg, nowUsed, nByte);
	system_mutex_enter(mem0.mutex);
	mem0.alarmCallback = xCallback;
	mem0.alarmArg = pArg;
}

/*
** Do a memory allocation with statistics and alarms. Assume the lock is already held.
*/
static int mallocWithAlarm(int n, void **pp)
{
	int nFull;
	void *p;
	assert(system_mutex_held(mem0.mutex));
	nFull = systemGlobalConfig.m.xRoundup(n);
	systemStatusSet(SYSTEM_MEMSTATUS_MALLOC_SIZE, n);
	if (mem0.alarmCallback != 0)
	{
		int nUsed = systemStatusValue(SYSTEM_MEMSTATUS_MEMORY_USED);
		if (nUsed+nFull >= mem0.alarmThreshold)
		{
			mem0.nearlyFull = 1;
			systemMallocAlarm(nFull);
		}
		else
			mem0.nearlyFull = 0;
	}
	p = systemGlobalConfig.m.xMalloc(nFull);
#ifdef SYSTEM_ENABLE_MEMORY_MANAGEMENT
	if (p == 0 && mem0.alarmCallback)
	{
		systemMallocAlarm(nFull);
		p = systemGlobalConfig.m.xMalloc(nFull);
	}
#endif
	if (p)
	{
		nFull = systemMallocSize(p);
		systemStatusAdd(SYSTEM_MEMSTATUS_MEMORY_USED, nFull);
		systemStatusAdd(SYSTEM_MEMSTATUS_MALLOC_COUNT, 1);
	}
	*pp = p;
	return nFull;
}

/*
** Allocate memory.  This routine is like system_malloc() except that it assumes the memory subsystem has already been initialized.
*/
void *systemMalloc(int n)
{
	void *p;
	if (n <= 0|| n >= 0x7fffff00)               /* IMP: R-65312-04917 */ 
		/* A memory allocation of a number of bytes which is near the maximum signed integer value might cause an integer overflow inside of the
		** xMalloc().  Hence we limit the maximum size to 0x7fffff00, giving 255 bytes of overhead.  SQLite itself will never use anything near
		** this amount.  The only way to reach the limit is with sqlite3_malloc() */
		p = 0;
	else if (systemGlobalConfig.bMemstat)
	{
		system_mutex_enter(mem0.mutex);
		mallocWithAlarm(n, &p);
		system_mutex_leave(mem0.mutex);
	}
	else
		p = systemGlobalConfig.m.xMalloc(n);
	assert(EIGHT_BYTE_ALIGNMENT(p));  /* IMP: R-04675-44850 */
	return p;
}

/*
** This version of the memory allocation is for use by the application. First make sure the memory subsystem is initialized, then do the
** allocation.
*/
void *system_malloc(int n)
{
#ifndef SYSTEM_OMIT_AUTOINIT
	if (system_initialize())
		return 0;
#endif
	return systemMalloc(n);
}

/*
** Each thread may only have a single outstanding allocation from xScratchMalloc().  We verify this constraint in the single-threaded
** case by setting scratchAllocOut to 1 when an allocation is outstanding clearing it when the allocation is freed.
*/
#if SYSTEM_THREADSAFE==0 && !defined(NDEBUG)
static int scratchAllocOut = 0;
#endif

/*
** Allocate memory that is to be used and released right away. This routine is similar to alloca() in that it is not intended
** for situations where the memory might be held long-term.  This routine is intended to get memory to old large transient data
** structures that would not normally fit on the stack of an embedded processor.
*/
void *systemScratchMalloc(int n)
{
	void *p;
	assert(n > 0);
	system_mutex_enter(mem0.mutex);
	if (mem0.nScratchFree && systemGlobalConfig.szScratch >= n)
	{
		p = mem0.pScratchFree;
		mem0.pScratchFree = mem0.pScratchFree->pNext;
		mem0.nScratchFree--;
		systemStatusAdd(SYSTEM_MEMSTATUS_SCRATCH_USED, 1);
		systemStatusSet(SYSTEM_MEMSTATUS_SCRATCH_SIZE, n);
		system_mutex_leave(mem0.mutex);
	}
	else
	{
		if (systemGlobalConfig.bMemstat)
		{
			systemStatusSet(SYSTEM_MEMSTATUS_SCRATCH_SIZE, n);
			n = mallocWithAlarm(n, &p);
			if (p)
				systemStatusAdd(SYSTEM_MEMSTATUS_SCRATCH_OVERFLOW, n);
			system_mutex_leave(mem0.mutex);
		}
		else
		{
			system_mutex_leave(mem0.mutex);
			p = systemGlobalConfig.m.xMalloc(n);
		}
		systemMemdebugSetType(p, MEMTYPE_SCRATCH);
	}
	assert(system_mutex_notheld(mem0.mutex));

#if SYSTEM_THREADSAFE==0 && !defined(NDEBUG)
	/* Verify that no more than two scratch allocations per thread are outstanding at one time.  (This is only checked in the
	** single-threaded case since checking in the multi-threaded case would be much more complicated.) */
	assert(scratchAllocOut <= 1);
	if (p)
		scratchAllocOut++;
#endif
	return p;
}
void sqlite3ScratchFree(void *p)
{
	if (p)
	{
#if SYSTEM_THREADSAFE==0 && !defined(NDEBUG)
		/* Verify that no more than two scratch allocation per thread is outstanding at one time.  (This is only checked in the
		** single-threaded case since checking in the multi-threaded case would be much more complicated.) */
		assert(scratchAllocOut >= 1 && scratchAllocOut <= 2);
		scratchAllocOut--;
#endif
		if (p >= systemGlobalConfig.pScratch && p < mem0.pScratchEnd)
		{
			/* Release memory from the SQLITE_CONFIG_SCRATCH allocation */
			ScratchFreeslot *pSlot;
			pSlot = (ScratchFreeslot*)p;
			system_mutex_enter(mem0.mutex);
			pSlot->pNext = mem0.pScratchFree;
			mem0.pScratchFree = pSlot;
			mem0.nScratchFree++;
			assert(mem0.nScratchFree<=sqlite3GlobalConfig.nScratch);
			systemStatusAdd(SYSTEM_MEMSTATUS_SCRATCH_USED, -1);
			system_mutex_leave(mem0.mutex);
		}
		else
		{
			/* Release memory back to the heap */
			assert(systemMemdebugHasType(p, MEMTYPE_SCRATCH));
			assert(systemMemdebugNoType(p, ~MEMTYPE_SCRATCH));
			systemMemdebugSetType(p, MEMTYPE_HEAP);
			if (systemGlobalConfig.bMemstat)
			{
				int iSize = systemMallocSize(p);
				system_mutex_enter(mem0.mutex);
				systemStatusAdd(SYSTEM_MEMSTATUS_SCRATCH_OVERFLOW, -iSize);
				systemStatusAdd(SYSTEM_MEMSTATUS_MEMORY_USED, -iSize);
				systemStatusAdd(SYSTEM_MEMSTATUS_MALLOC_COUNT, -1);
				systemGlobalConfig.m.xFree(p);
				system_mutex_leave(mem0.mutex);
			}
			else
				systemGlobalConfig.m.xFree(p);
		}
	}
}

/*
** TRUE if p is a lookaside memory allocation from db
*/
#ifndef SYSTEM_OMIT_LOOKASIDE
static int isLookaside(appContext *db, void *p)
{
	return p && p >= db->lookaside.pStart && p < db->lookaside.pEnd;
}
#else
#define isLookaside(A,B) 0
#endif

/*
** Return the size of a memory allocation previously obtained from systemMalloc() or system_malloc().
*/
int systemMallocSize(void *p)
{
	assert(systemMemdebugHasType(p, MEMTYPE_HEAP));
	assert(systemMemdebugNoType(p, MEMTYPE_DB));
	return systemGlobalConfig.m.xSize(p);
}
int systemCtxMallocSize(appContext *db, void *p)
{
	assert(db == 0 || system_mutex_held(db->mutex));
	if (db && isLookaside(db, p))
		return db->lookaside.sz;
	else
	{
		assert(systemMemdebugHasType(p, MEMTYPE_DB));
		assert(systemMemdebugHasType(p, MEMTYPE_LOOKASIDE | MEMTYPE_HEAP));
		assert(db != 0 || systemMemdebugNoType(p, MEMTYPE_LOOKASIDE));
		return systemGlobalConfig.m.xSize(p);
	}
}

/*
** Free memory previously obtained from sqlite3Malloc().
*/
void system_free(void *p)
{
	if (p == 0)
		return;  /* IMP: R-49053-54554 */
	assert(systemMemdebugNoType(p, MEMTYPE_DB));
	assert(systemMemdebugHasType(p, MEMTYPE_HEAP));
	if (systemGlobalConfig.bMemstat)
	{
		system_mutex_enter(mem0.mutex);
		systemStatusAdd(SYSTEM_MEMSTATUS_MEMORY_USED, -systemMallocSize(p));
		systemStatusAdd(SYSTEM_MEMSTATUS_MALLOC_COUNT, -1);
		systemGlobalConfig.m.xFree(p);
		system_mutex_leave(mem0.mutex);
	}
	else
		systemGlobalConfig.m.xFree(p);
}

/*
** Free memory that might be associated with a particular database connection.
*/
void systemCtxFree(appContext *db, void *p)
{
	assert(db == 0 || system_mutex_held(db->mutex));
	if (db)
	{
		if (db->pnBytesFreed)
		{
			*db->pnBytesFreed += systemCtxMallocSize(db, p);
			return;
		}
		if (isLookaside(db, p))
		{
			LookasideSlot *pBuf = (LookasideSlot*)p;
			pBuf->pNext = db->lookaside.pFree;
			db->lookaside.pFree = pBuf;
			db->lookaside.nOut--;
			return;
		}
	}
	assert(systemMemdebugHasType(p, MEMTYPE_DB));
	assert(systemMemdebugHasType(p, MEMTYPE_LOOKASIDE|MEMTYPE_HEAP));
	assert(db != 0 || systemMemdebugNoType(p, MEMTYPE_LOOKASIDE));
	systemMemdebugSetType(p, MEMTYPE_HEAP);
	system_free(p);
}

/*
** Change the size of an existing memory allocation
*/
void *systemRealloc(void *pOld, int nBytes)
{
	int nOld, nNew;
	void *pNew;
	if (pOld == 0)
		return systemMalloc(nBytes); /* IMP: R-28354-25769 */
	if (nBytes <= 0)
	{
		system_free(pOld); /* IMP: R-31593-10574 */
		return 0;
	}
	if (nBytes>=0x7fffff00)
		/* The 0x7ffff00 limit term is explained in comments on systemMalloc() */
		return 0;
	nOld = systemMallocSize(pOld);
	/* IMPLEMENTATION-OF: R-46199-30249 SQLite guarantees that the second argument to xRealloc is always a value returned by a prior call to
	** xRoundup. */
	nNew = systemGlobalConfig.m.xRoundup(nBytes);
	if (nOld == nNew)
		pNew = pOld;
	else if (systemGlobalConfig.bMemstat)
	{
		system_mutex_enter(mem0.mutex);
		systemStatusSet(SYSTEM_MEMSTATUS_MALLOC_SIZE, nBytes);
		if (systemStatusValue(SYSTEM_MEMSTATUS_MEMORY_USED)+nNew-nOld >= mem0.alarmThreshold)
			systemMallocAlarm(nNew-nOld);
		assert(systemMemdebugHasType(pOld, MEMTYPE_HEAP));
		assert(systemMemdebugNoType(pOld, ~MEMTYPE_HEAP));
		pNew = systemGlobalConfig.m.xRealloc(pOld, nNew);
		if (pNew == 0 && mem0.alarmCallback)
		{
			systemMallocAlarm(nBytes);
			pNew = systemGlobalConfig.m.xRealloc(pOld, nNew);
		}
		if (pNew)
		{
			nNew = systemMallocSize(pNew);
			systemStatusAdd(SYSTEM_MEMSTATUS_MEMORY_USED, nNew-nOld);
		}
		system_mutex_leave(mem0.mutex);
	}
	else
		pNew = systemGlobalConfig.m.xRealloc(pOld, nNew);
	assert(EIGHT_BYTE_ALIGNMENT(pNew)); /* IMP: R-04675-44850 */
	return pNew;
}

/*
** The public interface to sqlite3Realloc.  Make sure that the memory subsystem is initialized prior to invoking systemRealloc.
*/
void *system_realloc(void *pOld, int n)
{
#ifndef SYSTEM_OMIT_AUTOINIT
	if (system_initialize())
	  return 0;
#endif
	return systemRealloc(pOld, n);
}

/*
** Allocate and zero memory.
*/ 
void *systemMallocZero(int n)
{
	void *p = systemMalloc(n);
	if (p)
		memset(p, 0, n);
	return p;
}

/*
** Allocate and zero memory.  If the allocation fails, make the mallocFailed flag in the connection pointer.
*/
void *systemCtxMallocZero(appContext *db, int n)
{
	void *p = systemCtxMallocRaw(db, n);
	if (p)
		memset(p, 0, n);
	return p;
}

/*
** Allocate and zero memory.  If the allocation fails, make the mallocFailed flag in the connection pointer.
**
** If db!=0 and db->mallocFailed is true (indicating a prior malloc failure on the same database connection) then always return 0.
** Hence for a particular database connection, once malloc starts failing, it fails consistently until mallocFailed is reset.
** This is an important assumption.  There are many places in the code that do things like this:
**
**         int *a = (int*)systemCtxMallocRaw(db, 100);
**         int *b = (int*)systemCtxMallocRaw(db, 200);
**         if( b ) a[10] = 9;
**
** In other words, if a subsequent malloc (ex: "b") worked, it is assumed that all prior mallocs (ex: "a") worked too.
*/
void *systemCtxMallocRaw(appContext *db, int n)
{
	void *p;
	assert(db == 0 || system_mutex_held(db->mutex));
	assert(db == 0 || db->pnBytesFreed == 0);
#ifndef SYSTEM_OMIT_LOOKASIDE
	if (db)
	{
		LookasideSlot *pBuf;
		if (db->mallocFailed)
			return 0;
		if (db->lookaside.bEnabled && n <= db->lookaside.sz && (pBuf = db->lookaside.pFree) != 0)
		{
			db->lookaside.pFree = pBuf->pNext;
			db->lookaside.nOut++;
			if (db->lookaside.nOut>db->lookaside.mxOut)
				db->lookaside.mxOut = db->lookaside.nOut;
			return (void*)pBuf;
		}
	}
#else
	if (db && db->mallocFailed)
		return 0;
#endif
	p = systemMalloc(n);
	if (!p && db)
		db->mallocFailed = 1;
	systemMemdebugSetType(p, MEMTYPE_DB | (db && db->lookaside.bEnabled ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP));
	return p;
}

/*
** Resize the block of memory pointed to by p to n bytes. If the resize fails, set the mallocFailed flag in the connection object.
*/
void *systemCtxRealloc(appContext *db, void *p, int n)
{
	void *pNew = 0;
	assert(db != 0);
	assert(system_mutex_held(db->mutex));
	if (db->mallocFailed == 0)
	{
		if (p == 0)
			return systemCtxMallocRaw(db, n);
		if (isLookaside(db, p))
		{
			if (n <= db->lookaside.sz)
				return p;
			pNew = systemCtxMallocRaw(db, n);
			if (pNew)
			{
				memcpy(pNew, p, db->lookaside.sz);
				systemCtxFree(db, p);
			}
		}
		else
		{
			assert(systemMemdebugHasType(p, MEMTYPE_DB));
			assert(systemMemdebugHasType(p, MEMTYPE_LOOKASIDE|MEMTYPE_HEAP));
			systemMemdebugSetType(p, MEMTYPE_HEAP);
			pNew = system_realloc(p, n);
			if (!pNew)
			{
				systemMemdebugSetType(p, MEMTYPE_DB|MEMTYPE_HEAP);
				db->mallocFailed = 1;
			}
			systemMemdebugSetType(pNew, MEMTYPE_DB | (db->lookaside.bEnabled ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP));
		}
	}
	return pNew;
}

/*
** Attempt to reallocate p.  If the reallocation fails, then free p and set the mallocFailed flag in the database connection.
*/
void *systemCtxReallocOrFree(appContext *db, void *p, int n)
{
	void *pNew;
	pNew = systemCtxRealloc(db, p, n);
	if (!pNew)
		systemCtxFree(db, p);
	return pNew;
}

/*
** Make a copy of a string in memory obtained from sqliteMalloc(). These functions call sqlite3MallocRaw() directly instead of sqliteMalloc(). This
** is because when memory debugging is turned on, these two functions are called via macros that record the current file and line number in the
** ThreadData structure.
*/
char *systemCtxStrDup(appContext *db, const char *z)
{
	char *zNew;
	size_t n;
	if (z == 0)
		return 0;
	n = systemStrlen30(z) + 1;
	assert((n&0x7fffffff) == n);
	zNew = (char*)systemCtxMallocRaw(db, (int)n);
	if (zNew)
		memcpy(zNew, z, n);
	return zNew;
}
char *systemCtxStrNDup(appContext *db, const char *z, int n)
{
	char *zNew;
	if (z == 0)
		return 0;
	assert((n&0x7fffffff) == n);
	zNew = (char*)systemCtxMallocRaw(db, n+1);
	if (zNew)
	{
		memcpy(zNew, z, n);
		zNew[n] = 0;
	}
	return zNew;
}

/*
** Create a string from the zFromat argument and the va_list that follows. Store the string in memory obtained from systemMalloc() and make *pz
** point to that string.
*/
void systemSetString(char **pz, appContext *db, const char *zFormat, ...){
	va_list ap;
	char *z;
	va_start(ap, zFormat);
	z = systemCtxVMPrintf(db, zFormat, ap);
	va_end(ap);
	systemCtxFree(db, *pz);
	*pz = z;
}

/*
** This function must be called before exiting any API function (i.e. returning control to the user) that has called sqlite3_malloc or
** system_realloc.
**
** The returned value is normally a copy of the second argument to this function. However, if a malloc() failure has occurred since the previous
** invocation SYSTEM_NOMEM is returned instead. 
**
** If the first argument, db, is not NULL and a malloc() error has occurred, then the connection error-code (the value returned by sqlite3_errcode())
** is set to SYSTEM_NOMEM.
*/
int systemApiExit(appContext *db, int rc)
{
	/* If the db handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed 
	** is unsafe, as is the call to systemError(). */
	assert(!db || system_mutex_held(db->mutex));
	if (db && (db->mallocFailed || rc == SYSTEM_IOERR_NOMEM))
	{
		systemCtxError(db, SYSTEM_NOMEM, 0);
		db->mallocFailed = 0;
		rc = SYSTEM_NOMEM;
	}
	return rc & (db ? db->errMask : 0xff);
}

#ifndef SYSTEM_OMIT_BUILTIN_TEST

/*
** Global variables.
*/
typedef struct BenignMallocHooks BenignMallocHooks;
static SYSTEM_WSD struct BenignMallocHooks
{
	void (*xBenignBegin)(void);
	void (*xBenignEnd)(void);
} systemHooks = { 0, 0 };

/* The "wsdHooks" macro will resolve to the appropriate BenignMallocHooks structure.  If writable static data is unsupported on the target,
** we have to locate the state vector at run-time.  In the more common case where writable static data is supported, wsdHooks can refer directly
** to the "sqlite3Hooks" state vector declared above.
*/
#ifdef SYSTEM_OMIT_WSD
# define wsdHooksInit \
	BenignMallocHooks *x = &GLOBAL(BenignMallocHooks,systemHooks)
# define wsdHooks x[0]
#else
# define wsdHooksInit
# define wsdHooks systemHooks
#endif

/*
** Register hooks to call when sqlite3BeginBenignMalloc() and systemEndBenignMalloc() are called, respectively.
*/
void systemBenignMallocHooks(void(*xBenignBegin)(void), void(*xBenignEnd)(void))
{
	wsdHooksInit;
	wsdHooks.xBenignBegin = xBenignBegin;
	wsdHooks.xBenignEnd = xBenignEnd;
}

/*
** This (systemEndBenignMalloc()) is called by APPID code to indicate that subsequent malloc failures are benign. A call to systemEndBenignMalloc()
** indicates that subsequent malloc failures are non-benign.
*/
void systemBeginBenignMalloc(void)
{
	wsdHooksInit;
	if (wsdHooks.xBenignBegin)
		wsdHooks.xBenignBegin();
}
void systemEndBenignMalloc(void)
{
	wsdHooksInit;
	if (wsdHooks.xBenignEnd)
		wsdHooks.xBenignEnd();
}

#endif  /* SYSTEM_OMIT_BUILTIN_TEST */
