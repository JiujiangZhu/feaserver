#ifndef _SYSTEM_THREADING_H_
#define _SYSTEM_THREADING_H_
#include "SystemApi+Threading.h"

/*
** The SYSTEM_THREADSAFE macro must be defined as 0, 1, or 2.
** 0 means mutexes are permanently disable and the library is never threadsafe.
** 1 means the library is serialized which is the highest level of threadsafety.
** 2 means the libary is multithreaded - multiple threads can use APPID as long as no two threads try to use the same database connection at the same time.
*/
#if !defined(SYSTEM_THREADSAFE)
# define SYSTEM_THREADSAFE 1
#endif

// inlcudes
#include "System.Mutex.h"

/*
** An instance of the following structure is used to store the busy-handler callback for a given appcontext 
**
** The sqlite.busyHandler member of the sqlite struct contains the busy callback for the database handle. Each pager opened via the sqlite
** handle is passed a pointer to sqlite.busyHandler. The busy-handler callback is currently invoked only from within pager.c.
*/
typedef struct BusyHandler BusyHandler;
struct BusyHandler
{
	int (*xFunc)(void *,int);  /* The busy callback */
	void *pArg;                /* First arg to busy callback */
	int nBusy;                 /* Incremented with each busy call */
};

/*
** INTERNAL FUNCTION PROTOTYPES
*/
#ifndef SYSTEM_MUTEX_OMIT
system_mutex_methods const *systemDefaultMutex(void);
system_mutex_methods const *systemNoopMutex(void);
system_mutex *systemMutexAlloc(int);
int systemMutexInit(void);
int systemMutexEnd(void);
#endif

#endif /* _SYSTEM_THREADING_H_ */
