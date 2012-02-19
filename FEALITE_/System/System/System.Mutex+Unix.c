#include "System.h"
/*
** This file contains the C functions that implement mutexes for pthreads
**
** The code in this file is only used if we are compiling threadsafe under unix with pthreads.
**
** Note that this implementation requires a version of pthreads that supports recursive mutexes.
*/
#ifdef SYSTEM_MUTEX_UNIX
#include <pthread.h>

/*
** The system_mutex.id, system_mutex.nRef, and system_mutex.owner fields are necessary under two condidtions:  (1) Debug builds and (2) using
** home-grown mutexes.  Encapsulate these conditions into a single #define.
*/
#if defined(SYSTEM_DEBUG) || defined(SYSTEM_HOMEGROWN_RECURSIVE_MUTEX)
# define SYSTEM_MUTEX_NREF 1
#else
# define SYSTEM_MUTEX_NREF 0
#endif

/*
** Each recursive mutex is an instance of the following structure.
*/
struct system_mutex
{
	pthread_mutex_t mutex;     /* Mutex controlling the lock */
#if SYSTEM_MUTEX_NREF
	int id;                    /* Mutex type */
	volatile int nRef;         /* Number of entrances */
	volatile pthread_t owner;  /* Thread that is within this mutex */
	int trace;                 /* True to trace changes */
#endif
};
#if SYSTEM_MUTEX_NREF
#define SYSTEM_MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER, 0, 0, (pthread_t)0, 0 }
#else
#define SYSTEM_MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER }
#endif

/*
** The system_mutex_held() and system_mutex_notheld() routine are intended for use only inside assert() statements.  On some platforms,
** there might be race conditions that can cause these routines to deliver incorrect results.  In particular, if pthread_equal() is
** not an atomic operation, then these routines might delivery incorrect results.  On most platforms, pthread_equal() is a 
** comparison of two integers and is therefore atomic.  But we are told that HPUX is not such a platform.  If so, then these routines
** will not always work correctly on HPUX.
**
** On those platforms where pthread_equal() is not atomic, SQLitAPPID should be compiled without -DSYSTEM_DEBUG and with -DNDEBUG to
** make sure no assert() statements are evaluated and hence these
** routines are never called.
*/
#if !defined(NDEBUG) || defined(SYSTEM_DEBUG)
static int pthreadMutexHeld(system_mutex *p)
{
	return (p->nRef!=0 && pthread_equal(p->owner, pthread_self()));
}
static int pthreadMutexNotheld(system_mutex *p)
{
	return p->nRef==0 || pthread_equal(p->owner, pthread_self())==0;
}
#endif

/*
** Initialize and deinitialize the mutex subsystem.
*/
static int pthreadMutexInit(void) { return SYSTEM_OK; }
static int pthreadMutexEnd(void) { return SYSTEM_OK; }

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
** returns a different mutex on every call.  But for the static  mutex types, the same mutex is returned on every call that has
** the same type number.
*/
static system_mutex *pthreadMutexAlloc(int iType)
{
	static system_mutex staticMutexes[] = {
		SYSTEM_MUTEX_INITIALIZER,
		SYSTEM_MUTEX_INITIALIZER,
		SYSTEM_MUTEX_INITIALIZER,
		SYSTEM_MUTEX_INITIALIZER,
		SYSTEM_MUTEX_INITIALIZER,
		SYSTEM_MUTEX_INITIALIZER
	};
	system_mutex *p;
	switch (iType)
	{
		case SYSTEM_MUTEX_RECURSIVE:
		{
			p = (system_mutex*)systemMallocZero(sizeof(*p));
			if (p)
			{
#ifdef SYSTEM_HOMEGROWN_RECURSIVE_MUTEX
				/* If recursive mutexes are not available, we will have to build our own.  See below. */
				pthread_mutex_init(&p->mutex, 0);
#else
				/* Use a recursive mutex if it is available */
				pthread_mutexattr_t recursiveAttr;
				pthread_mutexattr_init(&recursiveAttr);
				pthread_mutexattr_settype(&recursiveAttr, PTHREAD_MUTEX_RECURSIVE);
				pthread_mutex_init(&p->mutex, &recursiveAttr);
				pthread_mutexattr_destroy(&recursiveAttr);
#endif
#if SYSTEM_MUTEX_NREF
				p->id = iType;
#endif
			}
			break;
		}
		case SYSTEM_MUTEX_FAST:
		{
			p = (system_mutex*)systemMallocZero(sizeof(*p));
			if (p)
			{
#if SYSTEM_MUTEX_NREF
				p->id = iType;
#endif
				pthread_mutex_init(&p->mutex, 0);
			}
			break;
		}
		default:
		{
			assert(iType-2 >= 0);
			assert(iType-2 < gArrayLength(staticMutexes));
			p = &staticMutexes[iType-2];
#if SYSTEM_MUTEX_NREF
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
static void pthreadMutexFree(system_mutex *p)
{
	assert(p->nRef==0);
	assert(p->id==SYSTEM_MUTEX_FAST || p->id==SYSTEM_MUTEX_RECURSIVE);
	pthread_mutex_destroy(&p->mutex);
	system_free(p);
}

/*
** The system_mutex_enter() and system_mutex_try() routines attempt to enter a mutex.  If another thread is already within the mutex,
** system_mutex_enter() will block and system_mutex_try() will return SYSTEM_BUSY.  The system_mutex_try() interface returns SYSTEM_OK
** upon successful entry.  Mutexes created using SYSTEM_MUTEX_RECURSIVE can be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static void pthreadMutexEnter(system_mutex *p)
{
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || pthreadMutexNotheld(p));
#ifdef SYSTEM_HOMEGROWN_RECURSIVE_MUTEX
	/*
	** If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	** is atomic - that it cannot be deceived into thinking self and p->owner are equal if p->owner changes between two values
	** that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	** separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	** are not met, then the mutexes will fail and problems will result.
	*/
	{
		pthread_t self = pthread_self();
		if (p->nRef>0 && pthread_equal(p->owner, self))
			p->nRef++;
		else
		{
			pthread_mutex_lock(&p->mutex);
			assert(p->nRef==0);
			p->owner = self;
			p->nRef = 1;
		}
	}
#else
	/* Use the built-in recursive mutexes if they are available. */
	pthread_mutex_lock(&p->mutex);
#if SYSTEM_MUTEX_NREF
	assert(p->nRef>0 || p->owner==0);
	p->owner = pthread_self();
	p->nRef++;
#endif
#endif
#ifdef SYSTEM_DEBUG
	if (p->trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->trace, p->nRef);
#endif
}

static int pthreadMutexTry(system_mutex *p)
{
	int rc;
	assert(p->id==SYSTEM_MUTEX_RECURSIVE || pthreadMutexNotheld(p));
#ifdef SYSTEM_HOMEGROWN_RECURSIVE_MUTEX
	/*
	** If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	** is atomic - that it cannot be deceived into thinking self and p->owner are equal if p->owner changes between two values
	** that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	** separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	** are not met, then the mutexes will fail and problems will result.
	*/
	{
		pthread_t self = pthread_self();
		if (p->nRef>0 && pthread_equal(p->owner, self))
		{
			p->nRef++;
			rc = SYSTEM_OK;
		}
		else if (pthread_mutex_trylock(&p->mutex)==0)
		{
			assert(p->nRef==0);
			p->owner = self;
			p->nRef = 1;
			rc = SYSTEM_OK;
		}
		else
			rc = SYSTEM_BUSY;
	}
#else
	/* Use the built-in recursive mutexes if they are available. */
	if (pthread_mutex_trylock(&p->mutex)==0)
	{
#if SYSTEM_MUTEX_NREF
		p->owner = pthread_self();
		p->nRef++;
#endif
		rc = SYSTEM_OK;
	}
	else
		rc = SYSTEM_BUSY;
#endif
#ifdef SYSTEM_DEBUG
	if (rc==SYSTEM_OK && p->trace)
		printf("enter mutex %p (%d) with nRef=%d\n", p, p->trace, p->nRef);
#endif
	return rc;
}

/*
** The system_mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  APPID will never do either.
*/
static void pthreadMutexLeave(system_mutex *p)
{
	assert(pthreadMutexHeld(p));
#if SYSTEM_MUTEX_NREF
	p->nRef--;
	if (p->nRef==0)
		p->owner = 0;
#endif
	assert(p->nRef==0 || p->id==SYSTEM_MUTEX_RECURSIVE);
#ifdef SYSTEM_HOMEGROWN_RECURSIVE_MUTEX
	if (p->nRef==0)
		pthread_mutex_unlock(&p->mutex);
#else
	pthread_mutex_unlock(&p->mutex);
#endif
#ifdef SYSTEM_DEBUG
	if (p->trace)
		printf("leave mutex %p (%d) with nRef=%d\n", p, p->trace, p->nRef);
#endif
}

system_mutex_methods const *systemDefaultMutex(void){
	static const system_mutex_methods sMutex = {
		pthreadMutexInit,
		pthreadMutexEnd,
		pthreadMutexAlloc,
		pthreadMutexFree,
		pthreadMutexEnter,
		pthreadMutexTry,
		pthreadMutexLeave,
#ifdef SYSTEM_DEBUG
		pthreadMutexHeld,
		pthreadMutexNotheld
#else
		0,
		0
#endif
	};
	return &sMutex;
}

#endif /* SYSTEM_MUTEX_UNIX */
