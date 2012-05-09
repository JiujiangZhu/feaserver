using System;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public partial class Pager
    {

#if !SQLITE_OMIT_WAL
        // This function is invoked once for each page that has already been written into the log file when a WAL transaction is rolled back.
        // Parameter iPg is the page number of said page. The pCtx argument is actually a pointer to the Pager structure.
        //
        // If page iPg is present in the cache, and has no outstanding references, it is discarded. Otherwise, if there are one or more outstanding
        // references, the page content is reloaded from the database. If the attempt to reload content from the database is required and fails, 
        // return an SQLite error code. Otherwise, SQLITE_OK.
        static int pagerUndoCallback(Pager pCtx, Pgno iPg)
        {
            var rc = SQLITE.OK;
            Pager pPager = (Pager)pCtx;
            PgHdr pPg;

            pPg = sqlite3PagerLookup(pPager, iPg);
            if (pPg)
            {
                if (sqlite3PcachePageRefcount(pPg) == 1)
                {
                    sqlite3PcacheDrop(pPg);
                }
                else
                {
                    rc = readDbPage(pPg);
                    if (rc == SQLITE.OK)
                    {
                        pPager.xReiniter(pPg);
                    }
                    sqlite3PagerUnref(pPg);
                }
            }
            // Normally, if a transaction is rolled back, any backup processes are updated as data is copied out of the rollback journal and into the
            // database. This is not generally possible with a WAL database, as rollback involves simply truncating the log file. Therefore, if one
            // or more frames have already been written to the log (and therefore also copied into the backup databases) as part of this transaction,
            // the backups must be restarted.
            sqlite3BackupRestart(pPager.pBackup);
            return rc;
        }

        // This function is called to rollback a transaction on a WAL database.
        static int pagerRollbackWal(Pager pPager)
        {
            int rc;                         /* Return Code */
            PgHdr pList;                   /* List of dirty pages to revert */

            /* For all pages in the cache that are currently dirty or have already been written (but not committed) to the log file, do one of the 
            ** following:
            **   + Discard the cached page (if refcount==0), or
            **   + Reload page content from the database (if refcount>0).
            */
            pPager.dbSize = pPager.dbOrigSize;
            rc = sqlite3WalUndo(pPager.pWal, pagerUndoCallback, pPager);
            pList = sqlite3PcacheDirtyList(pPager.pPCache);
            while (pList && rc == SQLITE.OK)
            {
                PgHdr pNext = pList.pDirty;
                rc = pagerUndoCallback(pPager, pList.pgno);
                pList = pNext;
            }

            return rc;
        }


        /*
        ** This function is a wrapper around sqlite3WalFrames(). As well as logging
        ** the contents of the list of pages headed by pList (connected by pDirty),
        ** this function notifies any active backup processes that the pages have
        ** changed.
        **
        ** The list of pages passed into this routine is always sorted by page number.
        ** Hence, if page 1 appears anywhere on the list, it will be the first page.
        */
        static int pagerWalFrames(Pager pPager, PgHdr pList, Pgno nTruncate, int isCommit, int syncFlags)
        {
            int rc;                         /* Return code */
#if DEBUG || (SQLITE_CHECK_PAGES)
            PgHdr p;                       /* For looping over pages */
#endif

            Debug.Assert(pPager.pWal);
#if SQLITE_DEBUG
/* Verify that the page list is in accending order */
for(p=pList; p && p->pDirty; p=p->pDirty){
assert( p->pgno < p->pDirty->pgno );
}
#endif

            if (isCommit)
            {
                /* If a WAL transaction is being committed, there is no point in writing
                ** any pages with page numbers greater than nTruncate into the WAL file.
                ** They will never be read by any client. So remove them from the pDirty
                ** list here. */
                PgHdr* p;
                PgHdr** ppNext = &pList;
                for (p = pList; (*ppNext = p); p = p->pDirty)
                {
                    if (p->pgno <= nTruncate) ppNext = &p->pDirty;
                }
                assert(pList);
            }


            if (pList->pgno == 1) pager_write_changecounter(pList);
            rc = sqlite3WalFrames(pPager.pWal,
            pPager.pageSize, pList, nTruncate, isCommit, syncFlags
            );
            if (rc == SQLITE_OK && pPager.pBackup)
            {
                PgHdr* p;
                for (p = pList; p; p = p->pDirty)
                {
                    sqlite3BackupUpdate(pPager.pBackup, p->pgno, (u8*)p->pData);
                }
            }

#if SQLITE_CHECK_PAGES
pList = sqlite3PcacheDirtyList(pPager.pPCache);
for(p=pList; p; p=p->pDirty){
pager_set_pagehash(p);
}
#endif

            return rc;
        }

        /*
        ** Begin a read transaction on the WAL.
        **
        ** This routine used to be called "pagerOpenSnapshot()" because it essentially
        ** makes a snapshot of the database at the current point in time and preserves
        ** that snapshot for use by the reader in spite of concurrently changes by
        ** other writers or checkpointers.
        */
        static int pagerBeginReadTransaction(Pager* pPager)
        {
            int rc;                         /* Return code */
            int changed = 0;                /* True if cache must be reset */

            assert(pagerUseWal(pPager));
            assert(pPager.eState == PAGER_OPEN || pPager.eState == PAGER_READER);

            /* sqlite3WalEndReadTransaction() was not called for the previous
            ** transaction in locking_mode=EXCLUSIVE.  So call it now.  If we
            ** are in locking_mode=NORMAL and EndRead() was previously called,
            ** the duplicate call is harmless.
            */
            sqlite3WalEndReadTransaction(pPager.pWal);

            rc = sqlite3WalBeginReadTransaction(pPager.pWal, &changed);
            if (rc != SQLITE_OK || changed)
            {
                pager_reset(pPager);
            }

            return rc;
        }
#endif


#if !SQLITE_OMIT_WAL
        /*
** Check if the *-wal file that corresponds to the database opened by pPager
** exists if the database is not empy, or verify that the *-wal file does
** not exist (by deleting it) if the database file is empty.
**
** If the database is not empty and the *-wal file exists, open the pager
** in WAL mode.  If the database is empty or if no *-wal file exists and
** if no error occurs, make sure Pager.journalMode is not set to
** PAGER_JOURNALMODE_WAL.
**
** Return SQLITE_OK or an error code.
**
** The caller must hold a SHARED lock on the database file to call this
** function. Because an EXCLUSIVE lock on the db file is required to delete 
** a WAL on a none-empty database, this ensures there is no race condition 
** between the xAccess() below and an xDelete() being executed by some 
** other connection.
*/
        static int pagerOpenWalIfPresent(Pager* pPager)
        {
            int rc = SQLITE_OK;
            Debug.Assert(pPager.eState == PAGER_OPEN);
            Debug.Assert(pPager.eLock >= SHARED_LOCK || pPager.noReadlock);

            if (!pPager.tempFile)
            {
                int isWal;                    /* True if WAL file exists */
                Pgno nPage;                   /* Size of the database file */

                rc = pagerPagecount(pPager, &nPage);
                if (rc) return rc;
                if (nPage == 0)
                {
                    rc = sqlite3OsDelete(pPager.pVfs, pPager.zWal, 0);
                    isWal = 0;
                }
                else
                {
                    rc = sqlite3OsAccess(
                    pPager.pVfs, pPager.zWal, SQLITE_ACCESS_EXISTS, &isWal
                    );
                }
                if (rc == SQLITE_OK)
                {
                    if (isWal)
                    {
                        testcase(sqlite3PcachePagecount(pPager.pPCache) == 0);
                        rc = sqlite3PagerOpenWal(pPager, 0);
                    }
                    else if (pPager.journalMode == PAGER_JOURNALMODE_WAL)
                    {
                        pPager.journalMode = PAGER_JOURNALMODE_DELETE;
                    }
                }
            }
            return rc;
        }
#endif


#if !SQLITE_OMIT_WAL
/*
** This function is called when the user invokes "PRAGMA wal_checkpoint",
** "PRAGMA wal_blocking_checkpoint" or calls the sqlite3_wal_checkpoint()
** or wal_blocking_checkpoint() API functions.
**
** Parameter eMode is one of SQLITE_CHECKPOINT_PASSIVE, FULL or RESTART.
*/
int sqlite3PagerCheckpoint(Pager *pPager, int eMode, int *pnLog, int *pnCkpt){
  int rc = SQLITE_OK;
  if( pPager.pWal ){
    rc = sqlite3WalCheckpoint(pPager.pWal, eMode,
        pPager.xBusyHandler, pPager.pBusyHandlerArg,
        pPager.ckptSyncFlags, pPager.pageSize, (u8 *)pPager.pTmpSpace,
        pnLog, pnCkpt
    );
  }
  return rc;
}

    int sqlite3PagerWalCallback(Pager *pPager){
return sqlite3WalCallback(pPager.pWal);
}

/*
** Return true if the underlying VFS for the given pager supports the
** primitives necessary for write-ahead logging.
*/
int sqlite3PagerWalSupported(Pager *pPager){
const sqlite3_io_methods *pMethods = pPager.fd->pMethods;
return pPager.exclusiveMode || (pMethods->iVersion>=2 && pMethods->xShmMap);
}

/*
** Attempt to take an exclusive lock on the database file. If a PENDING lock
** is obtained instead, immediately release it.
*/
static int pagerExclusiveLock(Pager *pPager){
int rc;                         /* Return code */

assert( pPager.eLock==SHARED_LOCK || pPager.eLock==EXCLUSIVE_LOCK );
rc = pagerLockDb(pPager, EXCLUSIVE_LOCK);
if( rc!=SQLITE_OK ){
/* If the attempt to grab the exclusive lock failed, release the
** pending lock that may have been obtained instead.  */
pagerUnlockDb(pPager, SHARED_LOCK);
}

return rc;
}

/*
** Call sqlite3WalOpen() to open the WAL handle. If the pager is in 
** exclusive-locking mode when this function is called, take an EXCLUSIVE
** lock on the database file and use heap-memory to store the wal-index
** in. Otherwise, use the normal shared-memory.
*/
static int pagerOpenWal(Pager *pPager){
int rc = SQLITE_OK;

assert( pPager.pWal==0 && pPager.tempFile==0 );
assert( pPager.eLock==SHARED_LOCK || pPager.eLock==EXCLUSIVE_LOCK || pPager.noReadlock);

/* If the pager is already in exclusive-mode, the WAL module will use 
** heap-memory for the wal-index instead of the VFS shared-memory 
** implementation. Take the exclusive lock now, before opening the WAL
** file, to make sure this is safe.
*/
if( pPager.exclusiveMode ){
rc = pagerExclusiveLock(pPager);
}

/* Open the connection to the log file. If this operation fails, 
** (e.g. due to malloc() failure), return an error code.
*/
if( rc==SQLITE_OK ){
rc = sqlite3WalOpen(pPager.pVfs, 
pPager.fd, pPager.zWal, pPager.exclusiveMode, &pPager.pWal
        pPager.journalSizeLimit, &pPager.pWal
);
}

return rc;
}


/*
** The caller must be holding a SHARED lock on the database file to call
** this function.
**
** If the pager passed as the first argument is open on a real database
** file (not a temp file or an in-memory database), and the WAL file
** is not already open, make an attempt to open it now. If successful,
** return SQLITE_OK. If an error occurs or the VFS used by the pager does 
** not support the xShmXXX() methods, return an error code. *pbOpen is
** not modified in either case.
**
** If the pager is open on a temp-file (or in-memory database), or if
** the WAL file is already open, set *pbOpen to 1 and return SQLITE_OK
** without doing anything.
*/
int sqlite3PagerOpenWal(
Pager *pPager,                  /* Pager object */
int *pbOpen                     /* OUT: Set to true if call is a no-op */
){
int rc = SQLITE_OK;             /* Return code */

assert( assert_pager_state(pPager) );
assert( pPager.eState==PAGER_OPEN   || pbOpen );
assert( pPager.eState==PAGER_READER || !pbOpen );
assert( pbOpen==0 || *pbOpen==0 );
assert( pbOpen!=0 || (!pPager.tempFile && !pPager.pWal) );

if( !pPager.tempFile && !pPager.pWal ){
if( !sqlite3PagerWalSupported(pPager) ) return SQLITE_CANTOPEN;

/* Close any rollback journal previously open */
sqlite3OsClose(pPager.jfd);

rc = pagerOpenWal(pPager);
if( rc==SQLITE_OK ){
pPager.journalMode = PAGER_JOURNALMODE_WAL;
pPager.eState = PAGER_OPEN;
}
}else{
*pbOpen = 1;
}

return rc;
}

/*
** This function is called to close the connection to the log file prior
** to switching from WAL to rollback mode.
**
** Before closing the log file, this function attempts to take an 
** EXCLUSIVE lock on the database file. If this cannot be obtained, an
** error (SQLITE_BUSY) is returned and the log connection is not closed.
** If successful, the EXCLUSIVE lock is not released before returning.
*/
int sqlite3PagerCloseWal(Pager *pPager){
int rc = SQLITE_OK;

assert( pPager.journalMode==PAGER_JOURNALMODE_WAL );

/* If the log file is not already open, but does exist in the file-system,
** it may need to be checkpointed before the connection can switch to
** rollback mode. Open it now so this can happen.
*/
if( !pPager.pWal ){
int logexists = 0;
rc = pagerLockDb(pPager, SHARED_LOCK);
if( rc==SQLITE_OK ){
rc = sqlite3OsAccess(
pPager.pVfs, pPager.zWal, SQLITE_ACCESS_EXISTS, &logexists
);
}
if( rc==SQLITE_OK && logexists ){
rc = pagerOpenWal(pPager);
}
}

/* Checkpoint and close the log. Because an EXCLUSIVE lock is held on
** the database file, the log and log-summary files will be deleted.
*/
if( rc==SQLITE_OK && pPager.pWal ){
rc = pagerExclusiveLock(pPager);
if( rc==SQLITE_OK ){
rc = sqlite3WalClose(pPager.pWal, pPager.ckptSyncFlags,
           pPager.pageSize, (u8*)pPager.pTmpSpace);
pPager.pWal = 0;
}
}
return rc;
}

#if SQLITE_HAS_CODEC
/*
** This function is called by the wal module when writing page content
** into the log file.
**
** This function returns a pointer to a buffer containing the encrypted
** page content. If a malloc fails, this function may return NULL.
*/
void sqlite3PagerCodec(PgHdr *pPg){
voidaData = 0;
CODEC2(pPg->pPager, pPg->pData, pPg->pgno, 6, return 0, aData);
return aData;
}
#endif
#endif
    }
}
