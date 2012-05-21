#ifndef _SYSTEM_CONFIG_H_
#define _SYSTEM_CONFIG_H_

/*
** Structure containing global configuration data for the APPID library.
**
** This structure also contains some state information.
*/
struct systemConfig
{
	int bMemstat;                     /* True to enable memory status */
	int bCoreMutex;                   /* True to enable core mutexing */
	int bFullMutex;                   /* True to enable full mutexing */
	int mxStrlen;                     /* Maximum string length */
	int szLookaside;                  /* Default lookaside buffer size */
	int nLookaside;                   /* Default lookaside buffer count */
	system_mem_methods m;            /* Low-level memory allocation interface */
	system_mutex_methods mutex;      /* Low-level mutex interface */
	system_pcache_methods pcache;    /* Low-level page-cache interface */
	void *pHeap;                      /* Heap storage space */
	int nHeap;                        /* Size of pHeap[] */
	int mnReq, mxReq;                 /* Min and max heap requests sizes */
	void *pScratch;                   /* Scratch memory */
	int szScratch;                    /* Size of each scratch buffer */
	int nScratch;                     /* Number of scratch buffers */
	void *pPage;                      /* Page cache memory */
	int szPage;                       /* Size of each page in pPage[] */
	int nPage;                        /* Number of pages in pPage[] */
	int mxParserStack;                /* maximum depth of the parser stack */
	int sharedCacheEnabled;           /* true if shared-cache mode enabled */
	/* The above might be initialized to non-zero.  The following need to always initially be zero, however. */
	int isInit;                       /* True after initialization has finished */
	int inProgress;                   /* True while initialization in progress */
	int isMutexInit;                  /* True after mutexes are initialized */
	int isMallocInit;                 /* True after malloc is initialized */
	int isPCacheInit;                 /* True after malloc is initialized */
	system_mutex *pInitMutex;        /* Mutex used by system_initialize() */
	int nRefInitMutex;                /* Number of users of pInitMutex */
	void (*xLog)(void*,int,const char*); /* Function for logging */
	void *pLogArg;                       /* First argument to xLog() */
};

#define systemGlobalConfig GLOBAL(struct systemConfig, systemConfig)

#ifndef SYSTEM_AMALGAMATION
extern SYSTEM_WSD struct systemConfig systemConfig;
#endif

#endif /* _SYSTEM_CONFIG_H_ */
