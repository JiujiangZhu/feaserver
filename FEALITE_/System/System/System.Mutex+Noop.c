/*
** This file contains the C functions that implement mutexes.
**
** This implementation in this file does not provide any mutual exclusion and is thus suitable for use only in applications
** that use APPID in a single thread.  The routines defined here are place-holders.  Applications can substitute working
** mutex routines at start-time using the
**
**     system_config(SYSTEM_CONFIG_MUTEX,...)
**
** interface.
**
** If compiled with SYSTEM_DEBUG, then additional logic is inserted that does error checking on mutexes to make sure they are being
** called correctly.
*/
#include "System.h"

#ifndef SYSTEM_MUTEX_OMIT

#ifndef SYSTEM_DEBUG
/*
** Stub routines for all mutex methods.
**
** This routines provide no mutual exclusion or error checking.
*/
static int noopMutexInit(void) { return SYSTEM_OK;  }
static int noopMutexEnd(void) { return SYSTEM_OK; }
static system_mutex *noopMutexAlloc(int id) {  UNUSED_PARAMETER(id); return (system_mutex*)8; }
static void noopMutexFree(system_mutex *p) { UNUSED_PARAMETER(p); return; }
static void noopMutexEnter(system_mutex *p) { UNUSED_PARAMETER(p); return; }
static int noopMutexTry(system_mutex *p) { UNUSED_PARAMETER(p); return SYSTEM_OK; }
static void noopMutexLeave(system_mutex *p) { UNUSED_PARAMETER(p); return; }

system_mutex_methods const *systemNoopMutex(void)
{
	static const system_mutex_methods sMutex = {
		noopMutexInit,
		noopMutexEnd,
		noopMutexAlloc,
		noopMutexFree,
		noopMutexEnter,
		noopMutexTry,
		noopMutexLeave,
		0,
		0,
	};
	return &sMutex;
}

#else
/*
** SYSTEM_DEBUG
** In this implementation, error checking is provided for testing and debugging purposes.  The mutexes still do not provide any
** mutual exclusion.
*/

/*
** The mutex object
*/
typedef struct system_debug_mutex
{
	int id;     /* The mutex type */
	int cnt;    /* Number of entries without a matching leave */
} system_debug_mutex;

/*
** The system_mutex_held() and system_mutex_notheld() routine are intended for use inside assert() statements.
*/
static int debugMutexHeld(system_mutex *pX)
{
	system_debug_mutex *p = (system_debug_mutex*)pX;
	return p==0 || p->cnt>0;
}
static int debugMutexNotheld(system_mutex *pX)
{
	system_debug_mutex *p = (system_debug_mutex*)pX;
	return p==0 || p->cnt==0;
}

/*
** Initialize and deinitialize the mutex subsystem.
*/
static int debugMutexInit(void)
{
	return SYSTEM_OK;
}
static int debugMutexEnd(void)
{
	return SYSTEM_OK;
}

/*
** The system_mutex_alloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL
** that means that a mutex could not be allocated. 
*/
static system_mutex *debugMutexAlloc(int id)
{
	static system_debug_mutex aStatic[6];
	system_debug_mutex *pNew = 0;
	switch (id)
	{
		case SYSTEM_MUTEX_FAST:
		case SYSTEM_MUTEX_RECURSIVE:
		{
			pNew = (system_debug_mutex*)systemMalloc(sizeof(*pNew));
			if (pNew)
			{
				pNew->id = id;
				pNew->cnt = 0;
			}
			break;
		}
		default:
		{
			assert(id-2 >= 0);
			assert(id-2 < (int)(sizeof(aStatic)/sizeof(aStatic[0])));
			pNew = &aStatic[id-2];
			pNew->id = id;
			break;
		}
	}
	return (system_mutex*)pNew;
}

/*
** This routine deallocates a previously allocated mutex.
*/
static void debugMutexFree(system_mutex *pX)
{
	system_debug_mutex *p = (system_debug_mutex*)pX;
	assert(p->cnt==0);
	assert(p->id==SYSTEM_MUTEX_FAST || p->id==SYSTEM_MUTEX_RECURSIVE);
	system_free(p);
}

/*
** The system_mutex_enter() and system_mutex_try() routines attempt to enter a mutex.  If another thread is already within the mutex,
** system_mutex_enter() will block and system_mutex_try() will return SYSTEM_BUSY.  The system_mutex_try() interface returns SYSTEM_OK
** upon successful entry.  Mutexes created using SYSTEM_MUTEX_RECURSIVE can be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static void debugMutexEnter(system_mutex *pX)
{
	system_debug_mutex *p = (system_debug_mutex*)pX;
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || debugMutexNotheld(pX));
	p->cnt++;
}
static int debugMutexTry(system_mutex *pX)
{
	system_debug_mutex *p = (system_debug_mutex*)pX;
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || debugMutexNotheld(pX));
	p->cnt++;
	return SYSTEM_OK;
}

/*
** The system_mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  APPID will never do either.
*/
static void debugMutexLeave(system_mutex *pX)
{
	system_debug_mutex *p = (system_debug_mutex*)pX;
	assert(debugMutexHeld(pX));
	p->cnt--;
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || debugMutexNotheld(pX));
}

system_mutex_methods const *systemNoopMutex(void)
{
	static const system_mutex_methods sMutex = {
		debugMutexInit,
		debugMutexEnd,
		debugMutexAlloc,
		debugMutexFree,
		debugMutexEnter,
		debugMutexTry,
		debugMutexLeave,
		debugMutexHeld,
		debugMutexNotheld
	};
	return &sMutex;
}
#endif /* SYSTEM_DEBUG */

/*
** If compiled with SYSTEM_MUTEX_NOOP, then the no-op mutex implementation is used regardless of the run-time threadsafety setting.
*/
#ifdef SYSTEM_MUTEX_NOOP
system_mutex_methods const *systemDefaultMutex(void)
{
	return systemNoopMutex();
}
#endif

#endif /* SYSTEM_MUTEX_OMIT */
