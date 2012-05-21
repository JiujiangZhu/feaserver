// This file contains the C functions that implement mutexes.
// This file contains code that is common across all mutex implementations.
#include "System.h"

#if defined(SYSTEM_DEBUG) && !defined(SYSTEM_MUTEX_OMIT)
// For debugging purposes, record when the mutex subsystem is initialized and uninitialized so that we can assert() if there is an attempt to
// allocate a mutex while the system is uninitialized.
static SYSTEM_WSD int mutexIsInit = 0;
#endif /* SYSTEM_DEBUG */

#ifndef SYSTEM_MUTEX_OMIT
// Initialize the mutex system.
int systemMutexInit(void)
{
	int rc = SYSTEM_OK;
	if (!systemGlobalConfig.mutex.xMutexAlloc)
	{
		// If the xMutexAlloc method has not been set, then the user did not install a mutex implementation via system_config() prior to 
		// system_initialize() being called. This block copies pointers to the default implementation into the sqlite3GlobalConfig structure.
		system_mutex_methods const *pFrom;
		system_mutex_methods *pTo = &systemGlobalConfig.mutex;
		if (systemGlobalConfig.bCoreMutex)
			pFrom = systemDefaultMutex();
		else
			pFrom = systemNoopMutex();
		memcpy(pTo, pFrom, offsetof(system_mutex_methods, xMutexAlloc));
		memcpy(&pTo->xMutexFree, &pFrom->xMutexFree, sizeof(*pTo) - offsetof(system_mutex_methods, xMutexFree));
		pTo->xMutexAlloc = pFrom->xMutexAlloc;
	}
	rc = systemGlobalConfig.mutex.xMutexInit();
#ifdef SYSTEM_DEBUG
	GLOBAL(int, mutexIsInit) = 1;
#endif
	return rc;
}

// Shutdown the mutex system. This call frees resources allocated by systemMutexInit().
int systemMutexEnd(void)
{
	int rc = SYSTEM_OK;
	if (systemGlobalConfig.mutex.xMutexEnd)
		rc = systemGlobalConfig.mutex.xMutexEnd();
#ifdef SYSTEM_DEBUG
	GLOBAL(int, mutexIsInit) = 0;
#endif
	return rc;
}

// Retrieve a pointer to a static mutex or allocate a new dynamic one.
system_mutex *system_mutex_alloc(int id)
{
#ifndef SYSTEM_OMIT_AUTOINIT
	if (system_initialize())
		return 0;
#endif
	return systemGlobalConfig.mutex.xMutexAlloc(id);
}

system_mutex *systemMutexAlloc(int id)
{
	if (!systemGlobalConfig.bCoreMutex)
		return 0;
	assert(GLOBAL(int, mutexIsInit));
	return systemGlobalConfig.mutex.xMutexAlloc(id);
}

// Free a dynamic mutex.
void system_mutex_free(system_mutex *p)
{
	if (p)
		systemGlobalConfig.mutex.xMutexFree(p);
}

// Obtain the mutex p. If some other thread already has the mutex, block until it can be obtained.
void system_mutex_enter(system_mutex *p)
{
	if (p)
		systemGlobalConfig.mutex.xMutexEnter(p);
}

// Obtain the mutex p. If successful, return SYSTEM_OK. Otherwise, if another thread holds the mutex and it cannot be obtained, return SYSTEM_BUSY.
int system_mutex_try(system_mutex *p)
{
	int rc = SYSTEM_OK;
	if (p)
		return systemGlobalConfig.mutex.xMutexTry(p);
	return rc;
}

// The system_mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior is undefined if the mutex 
// is not currently entered. If a NULL pointer is passed as an argument this function is a no-op.
void system_mutex_leave(system_mutex *p)
{
	if (p)
		systemGlobalConfig.mutex.xMutexLeave(p);
}

#ifndef NDEBUG
// The system_mutex_held() and system_mutex_notheld() routine are intended for use inside assert() statements.
int system_mutex_held(system_mutex *p)
{
	return ((p == 0) || systemGlobalConfig.mutex.xMutexHeld(p));
}
int system_mutex_notheld(system_mutex *p)
{
	return ((p == 0) || systemGlobalConfig.mutex.xMutexNotheld(p));
}
#endif
#endif /* SYSTEM_MUTEX_OMIT */
