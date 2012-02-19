/*
** This file contains OS interface code that is common to all architectures.
*/
#define _SYSTEM_OS_C_ 1
#include "System.h"
#undef _SYSTEM_OS_C_

/*
** The default APPID system_vfs implementations do not allocate memory (actually, System.OS+Unix.c allocates a small amount of memory
** from within OsOpen()), but some third-party implementations may. So we test the effects of a malloc() failing and the systemOsXXX()
** function returning SYSTEM_IOERR_NOMEM using the DO_OS_MALLOC_TEST macro.
**
** The following functions are instrumented for malloc() failure  testing:
**
**     systemOsOpen()
**     systemOsRead()
**     systemOsWrite()
**     systemOsSync()
**     systemOsLock()
**
*/
#if defined(SYSTEM_TEST)
int system_memdebug_vfs_oom_test = 1;
# define DO_OS_MALLOC_TEST(x) \
	if (system_memdebug_vfs_oom_test && (!x || !systemIsMemJournal(x))) { \
		void *pTstAlloc = systemMalloc(10); \
		if (!pTstAlloc) return SQLITE_IOERR_NOMEM; \
		system_free(pTstAlloc); \
	}
#else
# define DO_OS_MALLOC_TEST(x)
#endif

/*
** The following routines are convenience wrappers around methods of the system_file object.  This is mostly just syntactic sugar. All
** of this would be completely automatic if APPID were coded using C++ instead of plain old C.
*/
int systemOsClose(system_file *pId)
{
	int rc = SYSTEM_OK;
	if (pId->pMethods)
	{
		rc = pId->pMethods->xClose(pId);
		pId->pMethods = 0;
	}
	return rc;
}

int systemOsRead(system_file *id, void *pBuf, int amt, i64 offset) { DO_OS_MALLOC_TEST(id); return id->pMethods->xRead(id, pBuf, amt, offset); }
int systemOsWrite(system_file *id, const void *pBuf, int amt, i64 offset) { DO_OS_MALLOC_TEST(id); return id->pMethods->xWrite(id, pBuf, amt, offset); }
int systemOsTruncate(system_file *id, i64 size) { return id->pMethods->xTruncate(id, size); }
int systemOsSync(system_file *id, int flags) { DO_OS_MALLOC_TEST(id); return id->pMethods->xSync(id, flags); }
int systemOsFileSize(system_file *id, i64 *pSize) { DO_OS_MALLOC_TEST(id); return id->pMethods->xFileSize(id, pSize); }
int systemOsLock(system_file *id, int lockType) { DO_OS_MALLOC_TEST(id); return id->pMethods->xLock(id, lockType); }
int systemOsUnlock(system_file *id, int lockType) { return id->pMethods->xUnlock(id, lockType); }
int systemOsCheckReservedLock(system_file *id, int *pResOut) { DO_OS_MALLOC_TEST(id); return id->pMethods->xCheckReservedLock(id, pResOut); }
int systemOsFileControl(system_file *id, int op, void *pArg) { return id->pMethods->xFileControl(id, op, pArg); }
int systemOsSectorSize(system_file *id) { int (*xSectorSize)(system_file*) = id->pMethods->xSectorSize; return (xSectorSize ? xSectorSize(id) : SYSTEM_DEFAULT_SECTOR_SIZE); }
int systemOsDeviceCharacteristics(system_file *id) { return id->pMethods->xDeviceCharacteristics(id); }
int systemOsShmLock(system_file *id, int offset, int n, int flags) { return id->pMethods->xShmLock(id, offset, n, flags); }
void systemOsShmBarrier(system_file *id) { id->pMethods->xShmBarrier(id); }
int systemOsShmUnmap(system_file *id, int deleteFlag) { return id->pMethods->xShmUnmap(id, deleteFlag); }
int systemOsShmMap(system_file *id, int iPage, int pgsz, int bExtend, void volatile **pp ) { return id->pMethods->xShmMap(id, iPage, pgsz, bExtend, pp); } /* id: Database file handle, bExtend: True to extend file if necessary, pp: OUT: Pointer to mapping */

/*
** The next group of routines are convenience wrappers around the VFS methods.
*/
int systemOsOpen(system_vfs *pVfs, const char *zPath, system_file *pFile, int flags, int *pFlagsOut)
{
	int rc;
	DO_OS_MALLOC_TEST(0);
	/*
	** 0x87f3f is a mask of SYSTEM_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SYSTEM_OPEN_ flags (for example,
	** SYSTEM_OPEN_FULLMUTEX or SYSTEM_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
	*/
	rc = pVfs->xOpen(pVfs, zPath, pFile, flags & 0x87f3f, pFlagsOut);
	assert( rc==SQLITE_OK || pFile->pMethods==0 );
	return rc;
}
int systemOsDelete(system_vfs *pVfs, const char *zPath, int dirSync) { return pVfs->xDelete(pVfs, zPath, dirSync); }
int systemOsAccess(system_vfs *pVfs, const char *zPath, int flags, int *pResOut) { DO_OS_MALLOC_TEST(0); return pVfs->xAccess(pVfs, zPath, flags, pResOut); }
int systemOsFullPathname(system_vfs *pVfs, const char *zPath, int nPathOut, char *zPathOut) { zPathOut[0] = 0; return pVfs->xFullPathname(pVfs, zPath, nPathOut, zPathOut); }
#ifndef SYSTEM_OMIT_LOAD_EXTENSION
void *systemOsDlOpen(system_vfs *pVfs, const char *zPath) { return pVfs->xDlOpen(pVfs, zPath); }
void systemOsDlError(system_vfs *pVfs, int nByte, char *zBufOut) { pVfs->xDlError(pVfs, nByte, zBufOut); }
void (*systemOsDlSym(system_vfs *pVfs, void *pHdle, const char *zSym))(void) { return pVfs->xDlSym(pVfs, pHdle, zSym); }
void systemOsDlClose(system_vfs *pVfs, void *pHandle) { pVfs->xDlClose(pVfs, pHandle); }
#endif /* SYSTEM_OMIT_LOAD_EXTENSION */
int systemOsRandomness(system_vfs *pVfs, int nByte, char *zBufOut) { return pVfs->xRandomness(pVfs, nByte, zBufOut); }
int systemOsSleep(system_vfs *pVfs, int nMicro) { return pVfs->xSleep(pVfs, nMicro); }
int systemOsCurrentTimeInt64(system_vfs *pVfs, INT64_TYPE *pTimeOut)
{
	int rc;
	/*
	** IMPLEMENTATION-OF: R-49045-42493 APPID will use the xCurrentTimeInt64() method to get the current date and time if that method is available
	** (if iVersion is 2 or greater and the function pointer is not NULL) and will fall back to xCurrentTime() if xCurrentTimeInt64() is unavailable.
	*/
	if (pVfs->iVersion>=2 && pVfs->xCurrentTimeInt64)
		rc = pVfs->xCurrentTimeInt64(pVfs, pTimeOut);
	else
	{
		double r;
		rc = pVfs->xCurrentTime(pVfs, &r);
		*pTimeOut = (INT64_TYPE)(r*86400000.0);
	}
	return rc;
}

int systemOsOpenMalloc(system_vfs *pVfs, const char *zFile, system_file **ppFile, int flags, int *pOutFlags)
{
	int rc = SYSTEM_NOMEM;
	system_file *pFile;
	pFile = (system_file *)systemMalloc(pVfs->szOsFile);
	if (pFile)
	{
		rc = systemOsOpen(pVfs, zFile, pFile, flags, pOutFlags);
		if (rc!=SYSTEM_OK)
			system_free(pFile);
		else
			*ppFile = pFile;
	}
	return rc;
}

int systemOsCloseFree(system_file *pFile)
{
	int rc = SYSTEM_OK;
	assert(pFile);
	rc = systemOsClose(pFile);
	system_free(pFile);
	return rc;
}

/*
** This function is a wrapper around the OS specific implementation of system_os_init(). The purpose of the wrapper is to provide the
** ability to simulate a malloc failure, so that the handling of an error in system_os_init() by the upper layers can be tested.
*/
int systemOsInit(void)
{
	void *p = system_malloc(10);
	if (p==0)
		return SYSTEM_NOMEM;
	system_free(p);
	return system_os_init();
}

/*
** The list of all registered VFS implementations.
*/
static system_vfs * SYSTEM_WSD vfsList = 0;
#define vfsList GLOBAL(system_vfs*, vfsList)

/*
** Locate a VFS by name.  If no name is given, simply return the first VFS on the list.
*/
system_vfs *system_vfs_find(const char *zVfs)
{
	system_vfs *pVfs = 0;
#if SYSTEM_THREADSAFE
	system_mutex *mutex;
#endif
#ifndef SYSTEM_OMIT_AUTOINIT
	int rc = system_initialize();
	if (rc)
		return 0;
#endif
#if SYSTEM_THREADSAFE
	mutex = systemMutexAlloc(SYSTEM_MUTEX_STATIC_MASTER);
#endif
	system_mutex_enter(mutex);
	for (pVfs = vfsList; pVfs; pVfs=pVfs->pNext)
	{
		if (zVfs==0)
			break;
		if (strcmp(zVfs, pVfs->zName)==0)
			break;
	}
	system_mutex_leave(mutex);
	return pVfs;
}

/*
** Unlink a VFS from the linked list
*/
static void vfsUnlink(system_vfs *pVfs)
{
	assert(system_mutex_held(systemMutexAlloc(SQLITE_MUTEX_STATIC_MASTER)));
	if (pVfs==0) { /* No-op */ }
	else if (vfsList==pVfs)
		vfsList = pVfs->pNext;
	else if (vfsList)
	{
		system_vfs *p = vfsList;
		while (p->pNext && p->pNext!=pVfs)
			p = p->pNext;
		if (p->pNext==pVfs)
			p->pNext = pVfs->pNext;
	}
}

/*
** Register a VFS with the system.  It is harmless to register the same VFS multiple times.  The new VFS becomes the default if makeDflt is true.
*/
int system_vfs_register(system_vfs *pVfs, int makeDflt)
{
	system_mutex *mutex = 0;
#ifndef SYSTEM_OMIT_AUTOINIT
	int rc = system_initialize();
	if (rc)
		return rc;
#endif
	mutex = systemMutexAlloc(SYSTEM_MUTEX_STATIC_MASTER);
	system_mutex_enter(mutex);
	vfsUnlink(pVfs);
	if (makeDflt || vfsList==0)
	{
		pVfs->pNext = vfsList;
		vfsList = pVfs;
	}
	else
	{
		pVfs->pNext = vfsList->pNext;
		vfsList->pNext = pVfs;
	}
	assert(vfsList);
	system_mutex_leave(mutex);
	return SYSTEM_OK;
}

/*
** Unregister a VFS so that it is no longer accessible.
*/
int system_vfs_unregister(system_vfs *pVfs)
{
#if SYSTEM_THREADSAFE
	system_mutex *mutex = systemMutexAlloc(SYSTEM_MUTEX_STATIC_MASTER);
#endif
	system_mutex_enter(mutex);
	vfsUnlink(pVfs);
	system_mutex_leave(mutex);
	return SYSTEM_OK;
}
