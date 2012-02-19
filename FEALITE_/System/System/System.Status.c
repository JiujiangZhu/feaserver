/*
** This module implements the system_status() interface and related functionality.
*/
#include "System.h"
//#include "Vdbe.h"

/*
** Variables in which to record status information.
*/
typedef struct systemStatType systemStatType;
static SYSTEM_WSD struct sqlite3StatType
{
	int nowValue[10];         /* Current value */
	int mxValue[10];          /* Maximum value */
} systemStat = { {0,}, {0,} };


/* The "wsdStat" macro will resolve to the status information state vector.  If writable static data is unsupported on the target,
** we have to locate the state vector at run-time.  In the more common case where writable static data is supported, wsdStat can refer directly
** to the "sqlite3Stat" state vector declared above.
*/
#ifdef SYSTEM_OMIT_WSD
# define wsdStatInit  systemStatType *x = &GLOBAL(systemStatType,systemStat)
# define wsdStat x[0]
#else
# define wsdStatInit
# define wsdStat systemStat
#endif

/*
** Return the current value of a status parameter.
*/
int systemStatusValue(int op)
{
	wsdStatInit;
	assert(op >= 0 && op < gArrayLength(wsdStat.nowValue));
	return wsdStat.nowValue[op];
}

/*
** Add N to the value of a status record.  It is assumed that the caller holds appropriate locks.
*/
void systemStatusAdd(int op, int N)
{
	wsdStatInit;
	assert(op >= 0 && op < gArrayLength(wsdStat.nowValue));
	wsdStat.nowValue[op] += N;
	if (wsdStat.nowValue[op]>wsdStat.mxValue[op])
		wsdStat.mxValue[op] = wsdStat.nowValue[op];
}

/*
** Set the value of a status to X.
*/
void systemStatusSet(int op, int X)
{
	wsdStatInit;
	assert(op >= 0 && op < gArrayLength(wsdStat.nowValue));
	wsdStat.nowValue[op] = X;
	if (wsdStat.nowValue[op] > wsdStat.mxValue[op])
		wsdStat.mxValue[op] = wsdStat.nowValue[op];
}

/*
** Query status information.
**
** This implementation assumes that reading or writing an aligned 32-bit integer is an atomic operation.  If that assumption is not true,
** then this routine is not threadsafe.
*/
int system_status(int op, int *pCurrent, int *pHighwater, int resetFlag)
{
	wsdStatInit;
	if (op < 0 || op >= gArrayLength(wsdStat.nowValue))
		return SYSTEM_MISUSE_BKPT;
	*pCurrent = wsdStat.nowValue[op];
	*pHighwater = wsdStat.mxValue[op];
	if (resetFlag)
		wsdStat.mxValue[op] = wsdStat.nowValue[op];
	return SYSTEM_OK;
}

#if 0
/*
** Query status information for a single database connection
*/
int system_db_status(appContext *db, int op, int *pCurrent, int *pHighwater, int resetFlag){
	int rc = SYSTEM_OK;   /* Return code */
	system_mutex_enter(db->mutex);
	switch (op)
	{
		case SYSTEM_DBSTATUS_LOOKASIDE_USED:
		{
			*pCurrent = db->lookaside.nOut;
			*pHighwater = db->lookaside.mxOut;
			if (resetFlag)
				db->lookaside.mxOut = db->lookaside.nOut;
			break;
		}

		/* Return an approximation for the amount of memory currently used by all pagers associated with the given database connection.  The
		** highwater mark is meaningless and is returned as zero. */
		case SYSTEM_DBSTATUS_CACHE_USED:
		{
			int totalUsed = 0;
			int i;
			systemBtreeEnterAll(db);
			for (i = 0; i < db->nDb; i++ )
			{
				Btree *pBt = db->aDb[i].pBt;
				if (pBt)
				{
					Pager *pPager = systemBtreePager(pBt);
					totalUsed += systemPagerMemUsed(pPager);
				}
			}
			systemBtreeLeaveAll(db);
			*pCurrent = totalUsed;
			*pHighwater = 0;
			break;
		}

		/* *pCurrent gets an accurate estimate of the amount of memory used to store the schema for all databases (main, temp, and any ATTACHed
		** databases.  *pHighwater is set to zero. */
		case SYSTEM_DBSTATUS_SCHEMA_USED:
		{
			int i;                      /* Used to iterate through schemas */
			int nByte = 0;              /* Used to accumulate return value */
			db->pnBytesFreed = &nByte;
			for (i=0; i<db->nDb; i++)
			{
				Schema *pSchema = db->aDb[i].pSchema;
				if (ALWAYS(pSchema != 0))
				{
					HashElem *p;
					nByte += sqlite3GlobalConfig.m.xRoundup(sizeof(HashElem))*(pSchema->tblHash.count+pSchema->trigHash.count+pSchema->idxHash.count+pSchema->fkeyHash.count);
					nByte += sqlite3MallocSize(pSchema->tblHash.ht);
					nByte += sqlite3MallocSize(pSchema->trigHash.ht);
					nByte += sqlite3MallocSize(pSchema->idxHash.ht);
					nByte += sqlite3MallocSize(pSchema->fkeyHash.ht);
					for (p = systemHashFirst(&pSchema->trigHash); p; p = systemHashNext(p))
						systemDeleteTrigger(db, (Trigger*)sqliteHashData(p));
					for (p = systemHashFirst(&pSchema->tblHash); p; p = systemHashNext(p))
						systemDeleteTable(db, (Table*)systemHashData(p));
				}
			}
			db->pnBytesFreed = 0;
			*pHighwater = 0;
			*pCurrent = nByte;
			break;
		}

		/* *pCurrent gets an accurate estimate of the amount of memory used to store all prepared statements.
		** *pHighwater is set to zero. */
		case SYSTEM_DBSTATUS_STMT_USED:
		{
			struct Vdbe *pVdbe;         /* Used to iterate through VMs */
			int nByte = 0;              /* Used to accumulate return value */
			db->pnBytesFreed = &nByte;
			for (pVdbe = db->pVdbe; pVdbe; pVdbe = pVdbe->pNext)
				sqlite3VdbeDeleteObject(db, pVdbe);
			db->pnBytesFreed = 0;
			*pHighwater = 0;
			*pCurrent = nByte;
			break;
		}
		default:
		{
			rc = SYSTEM_ERROR;
		}
	}
	system_mutex_leave(db->mutex);
	return rc;
}
#endif