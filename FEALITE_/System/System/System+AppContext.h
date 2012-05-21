#ifndef _SYSTEM_APPCONTEXT_H_
#define _SYSTEM_APPCONTEXT_H_

/*
** Each connection is an instance of the following structure.
** The member variables sqlite.errCode, sqlite.zErrMsg and sqlite.zErrMsg16 store the most recent error code and, if applicable, string. The
** internal function sqlite3Error() is used to set these variables consistently.
*/
struct appContext
{
	system_vfs *pVfs;            /* OS Interface */
	//int nDb;                      /* Number of backends currently in use */
	//Db *aDb;                      /* All backends */
	int flags;                    /* Miscellaneous flags. See below */
	int openFlags;                /* Flags passed to system_vfs.xOpen() */
	int errCode;                  /* Most recent error code (SYSTEM_*) */
	int errMask;                  /* & result codes with this before returning */
	u8 autoCommit;                /* The auto-commit flag. */
	u8 temp_store;                /* 1: file 2: memory 0: default */
	u8 mallocFailed;              /* True if we have seen a malloc failure */
	u8 dfltLockMode;              /* Default locking-mode for attached dbs */
	signed char nextAutovac;      /* Autovac setting after VACUUM if >=0 */
	u8 suppressErr;               /* Do not issue error messages if true */
	int nextPagesize;             /* Pagesize after VACUUM if >0 */
	int nTable;                   /* Number of tables in the database */
	CollSeq *pDfltColl;           /* The default collating sequence (BINARY) */
	i64 lastRowid;                /* ROWID of most recent insert (see above) */
	u32 magic;                    /* Magic number for detect library misuse */
	int nChange;                  /* Value returned by system_changes() */
	int nTotalChange;             /* Value returned by system_total_changes() */
	system_mutex *mutex;         /* Connection mutex */
	int aLimit[SYSTEM_N_LIMIT];   /* Limits */
	struct systemInitInfo {      /* Information used during initialization */
	    int iDb;                    /* When back is being initialized */
	    int newTnum;                /* Rootpage of table being initialized */
	    u8 busy;                    /* TRUE if currently initializing */
	    u8 orphanTrigger;           /* Last statement is orphaned TEMP trigger */
	} init;
	int nExtension;               /* Number of loaded extensions */
	void **aExtension;            /* Array of shared library handles */
	struct Vdbe *pVdbe;           /* List of active virtual machines */
	int activeVdbeCnt;            /* Number of VDBEs currently executing */
	int writeVdbeCnt;             /* Number of active VDBEs that are writing */
	void (*xTrace)(void*,const char*);        /* Trace function */
	void *pTraceArg;                          /* Argument to the trace function */
	void (*xProfile)(void*,const char*,u64);  /* Profiling function */
	void *pProfileArg;                        /* Argument to profile function */
	void *pCommitArg;                 /* Argument to xCommitCallback() */   
	int (*xCommitCallback)(void*);    /* Invoked at every commit. */
	void *pRollbackArg;               /* Argument to xRollbackCallback() */   
	void (*xRollbackCallback)(void*); /* Invoked at every commit. */
	void *pUpdateArg;
	void (*xUpdateCallback)(void*, int, const char*, const char*, INT64_TYPE);
#ifndef SYSTEM_OMIT_WAL
	int (*xWalCallback)(void*, appContext*, const char*, int);
	void *pWalArg;
#endif
	void(*xCollNeeded)(void*, appContext*, int eTextRep, const char*);
	void(*xCollNeeded16)(void*, appContext*, int eTextRep, const void*);
	void *pCollNeededArg;
	system_value *pErr;          /* Most recent error message */
	char *zErrMsg;                /* Most recent error message (UTF-8 encoded) */
	char *zErrMsg16;              /* Most recent error message (UTF-16 encoded) */
	union {
		volatile int isInterrupted; /* True if system_interrupt has been called */
		double notUsed1;            /* Spacer */
	} u1;
	Lookaside lookaside;          /* Lookaside malloc configuration */
#ifndef SYSTEM_OMIT_AUTHORIZATION
  int (*xAuth)(void*,int,const char*,const char*,const char*,const char*); /* Access authorization function */
  void *pAuthArg;               /* 1st argument to the access auth function */
#endif
#ifndef SYSTEM_OMIT_PROGRESS_CALLBACK
	int (*xProgress)(void *);     /* The progress callback */
	void *pProgressArg;           /* Argument to the progress callback */
	int nProgressOps;             /* Number of opcodes for progress callback */
#endif
//#ifndef SYSTEM_OMIT_VIRTUALTABLE
//	Hash aModule;                 /* populated by system_create_module() */
//	Table *pVTab;                 /* vtab with active Connect/Create method */
//	VTable **aVTrans;             /* Virtual tables with open transactions */
//	int nVTrans;                  /* Allocated size of aVTrans */
//	VTable *pDisconnect;    /* Disconnect these in next system_prepare() */
//#endif
//	FuncDefHash aFunc;            /* Hash table of connection functions */
	Hash aCollSeq;                /* All collating sequences */
	BusyHandler busyHandler;      /* Busy callback */
	int busyTimeout;              /* Busy handler timeout, in msec */
//	Db aDbStatic[2];              /* Static space for the 2 default backends */
//	Savepoint *pSavepoint;        /* List of active savepoints */
	int nSavepoint;               /* Number of non-transaction savepoints */
	int nStatement;               /* Number of nested statement-transactions  */
	u8 isTransactionSavepoint;    /* True if the outermost savepoint is a TS */
	i64 nDeferredCons;            /* Net deferred constraints this transaction. */
	int *pnBytesFreed;            /* If not NULL, increment this in DbFree() */

#ifdef SYSTEM_ENABLE_UNLOCK_NOTIFY
  /* The following variables are all protected by the STATIC_MASTER  mutex, not by sqlite3.mutex. They are used by code in notify.c. 
  **
  ** When X.pUnlockConnection==Y, that means that X is waiting for Y to unlock so that it can proceed.
  **
  ** When X.pBlockingConnection==Y, that means that something that X tried tried to do recently failed with an SYSTEM_LOCKED error due to locks
  ** held by Y.
  */
	sqlite3 *pBlockingConnection; /* Connection that caused SYSTEM_LOCKED */
	sqlite3 *pUnlockConnection;           /* Connection to watch for unlock */
	void *pUnlockArg;                     /* Argument to xUnlockNotify */
	void (*xUnlockNotify)(void **, int);  /* Unlock notify callback */
	sqlite3 *pNextBlocked;        /* Next in list of all blocked connections */
#endif
};

void systemCtxError(appContext*, int, const char*,...);

char *systemCtxMPrintf(appContext*,const char*, ...);
char *systemCtxVMPrintf(appContext*,const char*, va_list);
char *systemCtxMAppendf(appContext*,char*,const char*,...);

int systemApiExit(appContext *db, int);

#endif /* _SYSTEM_APPCONTEXT_H_ */
