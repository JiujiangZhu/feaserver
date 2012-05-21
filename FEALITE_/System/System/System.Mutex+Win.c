#include "System.h"
/*
** This file contains the C functions that implement mutexes for win32
**
** The code in this file is only used if we are compiling multithreaded on a win32 system.
*/
#ifdef SYSTEM_MUTEX_WIN
/*
** Each recursive mutex is an instance of the following structure.
*/
struct system_mutex
{
	CRITICAL_SECTION mutex;    /* Mutex controlling the lock */
	int id;                    /* Mutex type */
#ifdef SYSTEM_DEBUG
	volatile int nRef;         /* Number of enterances */
	volatile DWORD owner;      /* Thread holding this mutex */
	int trace;                 /* True to trace changes */
#endif
};
#define SYSTEM_W32_MUTEX_INITIALIZER { 0 }
#ifdef SYSTEM_DEBUG
#define SYSTEM_MUTEX_INITIALIZER { SYSTEM_W32_MUTEX_INITIALIZER, 0, 0L, (DWORD)0, 0 }
#else
#define SYSTEM_MUTEX_INITIALIZER { SYSTEM_W32_MUTEX_INITIALIZER, 0 }
#endif

/*
** Return true (non-zero) if we are running under WinNT, Win2K, WinXP, or WinCE.  Return false (zero) for Win95, Win98, or WinME.
**
** Here is an interesting observation:  Win95, Win98, and WinME lack the LockFileEx() API.  But we can still statically link against that
** API as long as we don't call it win running Win95/98/ME.  A call to this routine is used to determine if the host is Win95/98/ME or
** WinNT/2K/XP so that we will know whether or not we can safely call the LockFileEx() API.
**
** mutexIsNT() is only used for the TryEnterCriticalSection() API call, which is only available if your application was compiled with 
** _WIN32_WINNT defined to a value >= 0x0400.  Currently, the only call to TryEnterCriticalSection() is #ifdef'ed out, so #ifdef 
** this out as well.
*/
#if 0
#if SYSTEM_OS_WINCE
# define mutexIsNT()  (1)
#else
static int mutexIsNT(void)
{
    static int osType = 0;
    if (osType==0)
	{
		OSVERSIONINFO sInfo;
		sInfo.dwOSVersionInfoSize = sizeof(sInfo);
		GetVersionEx(&sInfo);
		osType = (sInfo.dwPlatformId==VER_PLATFORM_WIN32_NT ? 2 : 1);
    }
    return (osType==2);
  }
#endif /* SYSTEM_OS_WINCE */
#endif

#ifndef SYSTEM_DEBUG
/*
** The system_mutex_held() and system_mutex_notheld() routine are intended for use only inside assert() statements.
*/
static int winMutexHeld(system_mutex *p)
{
	return (p->nRef!=0 && p->owner==GetCurrentThreadId());
}
static int winMutexNotheld2(system_mutex *p, DWORD tid)
{
	return (p->nRef==0 || p->owner!=tid);
}
static int winMutexNotheld(system_mutex *p)
{
	DWORD tid = GetCurrentThreadId(); 
	return winMutexNotheld2(p, tid);
}
#endif


/*
** Initialize and deinitialize the mutex subsystem.
*/
static system_mutex winMutex_staticMutexes[6] = {
	SYSTEM_MUTEX_INITIALIZER,
	SYSTEM_MUTEX_INITIALIZER,
	SYSTEM_MUTEX_INITIALIZER,
	SYSTEM_MUTEX_INITIALIZER,
	SYSTEM_MUTEX_INITIALIZER,
	SYSTEM_MUTEX_INITIALIZER
};
static int winMutex_isInit = 0;
/*
** As winMutexInit() and winMutexEnd() are called as part of the system_initialize and system_shutdown()
** processing, the "interlocked" magic is probably not strictly necessary.
*/
static long winMutex_lock = 0;

static int winMutexInit(void)
{
	/* The first to increment to 1 does actual initialization */
	if (InterlockedCompareExchange(&winMutex_lock, 1, 0)==0)
	{
		int i;
		for (i=0; i<gArrayLength(winMutex_staticMutexes); i++)
			InitializeCriticalSection(&winMutex_staticMutexes[i].mutex);
		winMutex_isInit = 1;
	}
	else
		/* Someone else is in the process of initing the static mutexes */
		while (!winMutex_isInit)
			Sleep(1);
	return SYSTEM_OK; 
}

static int winMutexEnd(void)
{ 
	/* The first to decrement to 0 does actual shutdown  (which should be the last to shutdown.) */
	if (InterlockedCompareExchange(&winMutex_lock, 0, 1)==1)
	{
		if (winMutex_isInit==1)
		{
			int i;
			for(i=0; i<gArrayLength(winMutex_staticMutexes); i++)
				DeleteCriticalSection(&winMutex_staticMutexes[i].mutex);
			winMutex_isInit = 0;
		}
	}
	return SYSTEM_OK; 
}

/*
** The system_mutex_alloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL
** that means that a mutex could not be allocated.  APPID will unwind its stack and return an error.  The argument
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
** </ul>
**
** The first two constants cause system_mutex_alloc() to create a new mutex.  The new mutex is recursive when SYSTEM_MUTEX_RECURSIVE
** is used but not necessarily so when SYSTEM_MUTEX_FAST is used. The mutex implementation does not need to make a distinction
** between SYSTEM_MUTEX_RECURSIVE and SYSTEM_MUTEX_FAST if it does not want to.  But APPID will only request a recursive mutex in
** cases where it really needs one.  If a faster non-recursive mutex implementation is available on the host platform, the mutex subsystem
** might return such a mutex in response to SYSTEM_MUTEX_FAST.
**
** The other allowed parameters to system_mutex_alloc() each return a pointer to a static preexisting mutex.  Six static mutexes are
** used by the current version of APPID.  Future versions of APPID may add additional static mutexes.  Static mutexes are for internal
** use by APPID only.  Applications that use APPID mutexes should use only the dynamic mutexes returned by SYSTEM_MUTEX_FAST or
** SYSTEM_MUTEX_RECURSIVE.
**
** Note that if one of the dynamic mutex parameters (SYSTEM_MUTEX_FAST or SYSTEM_MUTEX_RECURSIVE) is used then system_mutex_alloc()
** returns a different mutex on every call.  But for the static mutex types, the same mutex is returned on every call that has
** the same type number.
*/
static system_mutex *winMutexAlloc(int iType)
{
	system_mutex *p;
	switch (iType)
	{
		case SYSTEM_MUTEX_FAST:
		case SYSTEM_MUTEX_RECURSIVE:
		{
			p = (system_mutex*)systemMallocZero(sizeof(*p));
			if (p)
			{  
#ifdef SYSTEM_DEBUG
				p->id = iType;
#endif
				InitializeCriticalSection(&p->mutex);
			}
			break;
		}
		default:
		{
			assert(winMutex_isInit==1);
			assert(iType-2 >= 0);
			assert(iType-2 < gArrayLength(winMutex_staticMutexes));
			p = &winMutex_staticMutexes[iType-2];
#ifdef SYSTEM_DEBUG
			p->id = iType;
#endif
			break;
		}
	}
	return p;
}

/*
** This routine deallocates a previously allocated mutex.  APPID is careful to deallocate every
** mutex that it allocates.
*/
static void winMutexFree(system_mutex *p)
{
	assert(p);
	assert(p->nRef==0 && p->owner==0);
	assert(p->id==SYSTEM_MUTEX_FAST || p->id==SYSTEM_MUTEX_RECURSIVE);
	DeleteCriticalSection(&p->mutex);
	system_free(p);
}

/*
** The system_mutex_enter() and system_mutex_try() routines attempt to enter a mutex.  If another thread is already within the mutex,
** system_mutex_enter() will block and system_mutex_try() will return SYSTEM_BUSY.  The system_mutex_try() interface returns SYSTEM_OK
** upon successful entry.  Mutexes created using SYSTEM_MUTEX_RECURSIVE can be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static void winMutexEnter(system_mutex *p)
{
#ifdef SYSTEM_DEBUG
	DWORD tid = GetCurrentThreadId(); 
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || winMutexNotheld2(p, tid));
#endif
	EnterCriticalSection(&p->mutex);
#ifdef SYSTEM_DEBUG
	assert(p->nRef>0 || p->owner==0);
	p->owner = tid; 
	p->nRef++;
	if (p->trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->trace, p->nRef);
#endif
}

static int winMutexTry(system_mutex *p)
{
#ifndef NDEBUG
	DWORD tid = GetCurrentThreadId(); 
#endif
	int rc = SYSTEM_BUSY;
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || winMutexNotheld2(p, tid));
	/*
	** The system_mutex_try() routine is very rarely used, and when it is used it is merely an optimization.  So it is OK for it to always
	** fail.  
	**
	** The TryEnterCriticalSection() interface is only available on WinNT. And some windows compilers complain if you try to use it without
	** first doing some #defines that prevent APPID from building on Win98. For that reason, we will omit this optimization for now.
	*/
#if 0
	if (mutexIsNT() && TryEnterCriticalSection(&p->mutex))
	{
		p->owner = tid;
		p->nRef++;
		rc = SYSTEM_OK;
	}
#else
	UNUSED_PARAMETER(p);
#endif
#ifdef SYSTEM_DEBUG
	if (rc==SYSTEM_OK && p->trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->trace, p->nRef);
#endif
	return rc;
}

/*
** The system_mutex_leave() routine exits a mutex that was
** previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or
** is not currently allocated.  APPID will never do either.
*/
static void winMutexLeave(system_mutex *p){
#ifndef NDEBUG
	DWORD tid = GetCurrentThreadId();
	assert(p->nRef>0);
	assert(p->owner==tid);
	p->nRef--;
	if (p->nRef==0)
		p->owner = 0;
	assert(p->nRef==0 || p->id==SYSTEM_MUTEX_RECURSIVE);
#endif
	LeaveCriticalSection(&p->mutex);
#ifdef SYSTEM_DEBUG
	if (p->trace)
		printf("leave mutex %p (%d) with nRef=%d\n", p, p->trace, p->nRef);
#endif
}

system_mutex_methods const *systemDefaultMutex(void)
{
	static const system_mutex_methods sMutex = {
		winMutexInit,
		winMutexEnd,
		winMutexAlloc,
		winMutexFree,
		winMutexEnter,
		winMutexTry,
		winMutexLeave,
#ifdef SYSTEM_DEBUG
		winMutexHeld,
		winMutexNotheld
#else
		0,
		0
#endif
	};
	return &sMutex;
}
#endif /* SYSTEM_MUTEX_WIN */
