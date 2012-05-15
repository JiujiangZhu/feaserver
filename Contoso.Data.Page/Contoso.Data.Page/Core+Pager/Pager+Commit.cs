﻿using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal SQLITE sqlite3PagerCommitPhaseOne(string zMaster, bool noSync)
        {
            var rc = SQLITE.OK;
            Debug.Assert(eState == PAGER.WRITER_LOCKED || eState == PAGER.WRITER_CACHEMOD || eState == PAGER.WRITER_DBMOD || eState == PAGER.ERROR);
            Debug.Assert(assert_pager_state());
            // If a prior error occurred, report that error again.
            if (Check.NEVER(errCode != 0))
                return errCode;
            PAGERTRACE("DATABASE SYNC: File={0} zMaster={1} nSize={2}", zFilename, zMaster, dbSize);
            // If no database changes have been made, return early.
            if (eState < PAGER.WRITER_CACHEMOD)
                return SQLITE.OK;
            if (
#if SQLITE_OMIT_MEMORYDB
0 != MEMDB
#else
0 != memDb
#endif
)
            {
                // If this is an in-memory db, or no pages have been written to, or this function has already been called, it is mostly a no-op.  However, any
                // backup in progress needs to be restarted.
                if (pBackup != null)
                    pBackup.sqlite3BackupRestart();
            }
            else
            {
                if (pagerUseWal())
                {
                    var pList = pPCache.sqlite3PcacheDirtyList();
                    PgHdr pPageOne = null;
                    if (pList == null)
                    {
                        // Must have at least one page for the WAL commit flag.
                        rc = sqlite3PagerGet(1, ref pPageOne);
                        pList = pPageOne;
                        pList.pDirty = null;
                    }
                    Debug.Assert(rc == SQLITE.OK);
                    if (Check.ALWAYS(pList))
                        rc = pagerWalFrames(pList, dbSize, 1, (fullSync ? syncFlags : 0));
                    sqlite3PagerUnref(pPageOne);
                    if (rc == SQLITE.OK)
                        pPCache.sqlite3PcacheCleanAll();
                }
                else
                {
                    // The following block updates the change-counter. Exactly how it does this depends on whether or not the atomic-update optimization
                    // was enabled at compile time, and if this transaction meets the runtime criteria to use the operation:
                    //    * The file-system supports the atomic-write property for blocks of size page-size, and
                    //    * This commit is not part of a multi-file transaction, and
                    //    * Exactly one page has been modified and store in the journal file.
                    // If the optimization was not enabled at compile time, then the pager_incr_changecounter() function is called to update the change
                    // counter in 'indirect-mode'. If the optimization is compiled in but is not applicable to this transaction, call sqlite3JournalCreate()
                    // to make sure the journal file has actually been created, then call pager_incr_changecounter() to update the change-counter in indirect
                    // mode.
                    // Otherwise, if the optimization is both enabled and applicable, then call pager_incr_changecounter() to update the change-counter
                    // in 'direct' mode. In this case the journal file will never be created for this transaction.
#if SQLITE_ENABLE_ATOMIC_WRITE
                    PgHdr pPg;
                    Debug.Assert(this.jfd.isOpen || this.journalMode == JOURNALMODE.OFF || this.journalMode == JOURNALMODE.WAL);
                    if (!zMaster && this.jfd.isOpen
                        && this.journalOff == jrnlBufferSize(this)
                        && this.dbSize >= this.dbOrigSize
                        && (0 == (pPg = sqlite3PcacheDirtyList(this.pPCache)) || 0 == pPg.pDirty))
                        // Update the db file change counter via the direct-write method. The following call will modify the in-memory representation of page 1
                        // to include the updated change counter and then write page 1 directly to the database file. Because of the atomic-write
                        // property of the host file-system, this is safe.
                        rc = pager_incr_changecounter(this, 1);
                    else
                    {
                        rc = sqlite3JournalCreate(this.jfd);
                        if (rc == SQLITE.OK)
                            rc = pager_incr_changecounter(this, 0);
                    }
#else
                    rc = pager_incr_changecounter(false);
#endif
                    if (rc != SQLITE.OK)
                        goto commit_phase_one_exit;
                    // If this transaction has made the database smaller, then all pages being discarded by the truncation must be written to the journal
                    // file. This can only happen in auto-vacuum mode.
                    // Before reading the pages with page numbers larger than the current value of Pager.dbSize, set dbSize back to the value
                    // that it took at the start of the transaction. Otherwise, the calls to sqlite3PagerGet() return zeroed pages instead of
                    // reading data from the database file.
#if !SQLITE_OMIT_AUTOVACUUM
                    if (dbSize < dbOrigSize && journalMode != JOURNALMODE.OFF)
                    {
                        var iSkip = PAGER_MJ_PGNO(this); // Pending lock page
                        var lastDbSize = this.dbSize;       // Database image size
                        this.dbSize = dbOrigSize;
                        for (Pgno i = lastDbSize + 1; i <= dbOrigSize; i++)
                            if (0 == pInJournal.sqlite3BitvecTest(i) && i != iSkip)
                            {
                                PgHdr pPage = null;             // Page to journal
                                rc = sqlite3PagerGet(i, ref pPage);
                                if (rc != SQLITE.OK)
                                    goto commit_phase_one_exit;
                                rc = sqlite3PagerWrite(pPage);
                                sqlite3PagerUnref(pPage);
                                if (rc != SQLITE.OK)
                                    goto commit_phase_one_exit;
                            }
                        this.dbSize = lastDbSize;
                    }
#endif

                    // Write the master journal name into the journal file. If a master journal file name has already been written to the journal file,
                    // or if zMaster is NULL (no master journal), then this call is a no-op.
                    rc = writeMasterJournal(zMaster);
                    if (rc != SQLITE.OK)
                        goto commit_phase_one_exit;
                    // Sync the journal file and write all dirty pages to the database. If the atomic-update optimization is being used, this sync will not 
                    // create the journal file or perform any real IO.
                    // Because the change-counter page was just modified, unless the atomic-update optimization is used it is almost certain that the
                    // journal requires a sync here. However, in locking_mode=exclusive on a system under memory pressure it is just possible that this is 
                    // not the case. In this case it is likely enough that the redundant xSync() call will be changed to a no-op by the OS anyhow. 
                    rc = syncJournal(0);
                    if (rc != SQLITE.OK)
                        goto commit_phase_one_exit;
                    rc = pager_write_pagelist(pPCache.sqlite3PcacheDirtyList());
                    if (rc != SQLITE.OK)
                    {
                        Debug.Assert(rc != SQLITE.IOERR_BLOCKED);
                        goto commit_phase_one_exit;
                    }
                    pPCache.sqlite3PcacheCleanAll();
                    // If the file on disk is not the same size as the database image, then use pager_truncate to grow or shrink the file here.
                    if (this.dbSize != dbFileSize)
                    {
                        var nNew = (Pgno)(this.dbSize - (this.dbSize == PAGER_MJ_PGNO(this) ? 1 : 0));
                        Debug.Assert(this.eState >= PAGER.WRITER_DBMOD);
                        rc = pager_truncate(nNew);
                        if (rc != SQLITE.OK)
                            goto commit_phase_one_exit;
                    }
                    // Finally, sync the database file.
                    if (!noSync)
                        rc = sqlite3PagerSync();
                    SysEx.IOTRACE("DBSYNC {0:x}", this.GetHashCode());
                }
            }
        commit_phase_one_exit:
            if (rc == SQLITE.OK && !pagerUseWal())
                this.eState = PAGER.WRITER_FINISHED;
            return rc;
        }

        internal SQLITE sqlite3PagerCommitPhaseTwo()
        {
            var rc = SQLITE.OK;
            // This routine should not be called if a prior error has occurred. But if (due to a coding error elsewhere in the system) it does get
            // called, just return the same error code without doing anything. */
            if (Check.NEVER(this.errCode))
                return this.errCode;
            Debug.Assert(this.eState == PAGER.WRITER_LOCKED || this.eState == PAGER.WRITER_FINISHED || (pagerUseWal() && this.eState == PAGER.WRITER_CACHEMOD));
            Debug.Assert(assert_pager_state());
            // An optimization. If the database was not actually modified during this transaction, the pager is running in exclusive-mode and is
            // using persistent journals, then this function is a no-op.
            // The start of the journal file currently contains a single journal header with the nRec field set to 0. If such a journal is used as
            // a hot-journal during hot-journal rollback, 0 changes will be made to the database file. So there is no need to zero the journal
            // header. Since the pager is in exclusive mode, there is no need to drop any locks either.
            if (this.eState == PAGER.WRITER_LOCKED && this.exclusiveMode && this.journalMode == JOURNALMODE.PERSIST)
            {
                Debug.Assert(this.journalOff == JOURNAL_HDR_SZ(this) || 0 == this.journalOff);
                this.eState = PAGER.READER;
                return SQLITE.OK;
            }
            PAGERTRACE("COMMIT {0}", PAGERID(this));
            rc = pager_end_transaction(this.setMaster);
            return pager_error(rc);
        }

        internal SQLITE sqlite3PagerRollback()
        {
            var rc = SQLITE.OK;
            PAGERTRACE("ROLLBACK {0}", PAGERID(this));
            // PagerRollback() is a no-op if called in READER or OPEN state. If the pager is already in the ERROR state, the rollback is not 
            // attempted here. Instead, the error code is returned to the caller.
            Debug.Assert(assert_pager_state());
            if (this.eState == PAGER.ERROR)
                return this.errCode;
            if (this.eState <= PAGER.READER)
                return SQLITE.OK;
            if (pagerUseWal())
            {
                rc = sqlite3PagerSavepoint(SAVEPOINT.ROLLBACK, -1);
                var rc2 = pager_end_transaction(this.setMaster);
                if (rc == SQLITE.OK)
                    rc = rc2;
                rc = pager_error(rc);
            }
            else if (!this.jfd.isOpen || this.eState == PAGER.WRITER_LOCKED)
            {
                var eState = this.eState;
                rc = pager_end_transaction(0);
                if (
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == this.memDb
#endif
 && eState > PAGER.WRITER_LOCKED)
                {
                    // This can happen using journal_mode=off. Move the pager to the error  state to indicate that the contents of the cache may not be trusted.
                    // Any active readers will get SQLITE_ABORT.
                    this.errCode = SQLITE.ABORT;
                    this.eState = PAGER.ERROR;
                    return rc;
                }
            }
            else
                rc = pager_playback(0);
            Debug.Assert(this.eState == PAGER.READER || rc != SQLITE.OK);
            Debug.Assert(rc == SQLITE.OK || rc == SQLITE.FULL || ((int)rc & 0xFF) == (int)SQLITE.IOERR);
            // If an error occurs during a ROLLBACK, we can no longer trust the pager cache. So call pager_error() on the way out to make any error persistent.
            return pager_error(rc);
        }

#if !SQLITE_OMIT_AUTOVACUUM
        internal SQLITE sqlite3PagerMovepage(DbPage pPg, Pgno pgno, int isCommit)
        {
            Debug.Assert(pPg.nRef > 0);
            Debug.Assert(this.eState == PAGER.WRITER_CACHEMOD || this.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state());
            // In order to be able to rollback, an in-memory database must journal the page we are moving from.
            var rc = SQLITE.OK;
            if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
this.memDb != 0
#endif
)
            {
                rc = sqlite3PagerWrite(pPg);
                if (rc != SQLITE.OK)
                    return rc;
            }
            PgHdr pPgOld;                // The page being overwritten.
            uint needSyncPgno = 0;       // Old value of pPg.pgno, if sync is required
            Pgno origPgno;               // The original page number
            // If the page being moved is dirty and has not been saved by the latest savepoint, then save the current contents of the page into the
            // sub-journal now. This is required to handle the following scenario:
            //   BEGIN;
            //     <journal page X, then modify it in memory>
            //     SAVEPOINT one;
            //       <Move page X to location Y>
            //     ROLLBACK TO one;
            // If page X were not written to the sub-journal here, it would not be possible to restore its contents when the "ROLLBACK TO one"
            // statement were is processed.
            // subjournalPage() may need to allocate space to store pPg.pgno into one or more savepoint bitvecs. This is the reason this function
            // may return SQLITE_NOMEM.
            if ((pPg.flags & PgHdr.PGHDR.DIRTY) != 0 && subjRequiresPage(pPg) && SQLITE.OK != (rc = subjournalPage(pPg)))
                return rc;
            PAGERTRACE("MOVE {0} page {1} (needSync={2}) moves to {3}",
            PAGERID(this), pPg.pgno, (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0, pgno);
            SysEx.IOTRACE("MOVE {0} {1} {2}", this.GetHashCode(), pPg.pgno, pgno);
            // If the journal needs to be sync()ed before page pPg.pgno can be written to, store pPg.pgno in local variable needSyncPgno.
            // If the isCommit flag is set, there is no need to remember that the journal needs to be sync()ed before database page pPg.pgno
            // can be written to. The caller has already promised not to write to it.
            if (((pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0) && 0 == isCommit)
            {
                needSyncPgno = pPg.pgno;
                Debug.Assert(pageInJournal(pPg) || pPg.pgno > this.dbOrigSize);
                Debug.Assert((pPg.flags & PgHdr.PGHDR.DIRTY) != 0);
            }
            // If the cache contains a page with page-number pgno, remove it from its hash chain. Also, if the PGHDR_NEED_SYNC was set for
            // page pgno before the 'move' operation, it needs to be retained for the page moved there.
            pPg.flags &= ~PgHdr.PGHDR.NEED_SYNC;
            pPgOld = pager_lookup(pgno);
            Debug.Assert(null == pPgOld || pPgOld.nRef == 1);
            if (pPgOld != null)
            {
                pPg.flags |= (pPgOld.flags & PgHdr.PGHDR.NEED_SYNC);
                if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
this.memDb != 0
#endif
)
                    // Do not discard pages from an in-memory database since we might need to rollback later.  Just move the page out of the way.
                    PCache.sqlite3PcacheMove(pPgOld, this.dbSize + 1);
                else
                    PCache.sqlite3PcacheDrop(pPgOld);
            }
            origPgno = pPg.pgno;
            PCache.sqlite3PcacheMove(pPg, pgno);
            PCache.sqlite3PcacheMakeDirty(pPg);
            // For an in-memory database, make sure the original page continues to exist, in case the transaction needs to roll back.  Use pPgOld
            // as the original page since it has already been allocated.
            if (
#if SQLITE_OMIT_MEMORYDB
0!=MEMDB
#else
0 != this.memDb
#endif
)
            {
                Debug.Assert(pPgOld);
                PCache.sqlite3PcacheMove(pPgOld, origPgno);
                sqlite3PagerUnref(pPgOld);
            }
            if (needSyncPgno != 0)
            {
                // If needSyncPgno is non-zero, then the journal file needs to be sync()ed before any data is written to database file page needSyncPgno.
                // Currently, no such page exists in the page-cache and the "is journaled" bitvec flag has been set. This needs to be remedied by
                // loading the page into the pager-cache and setting the PGHDR_NEED_SYNC flag.
                // If the attempt to load the page into the page-cache fails, (due to a malloc() or IO failure), clear the bit in the pInJournal[]
                // array. Otherwise, if the page is loaded and written again in this transaction, it may be written to the database file before
                // it is synced into the journal file. This way, it may end up in the journal file twice, but that is not a problem.
                PgHdr pPgHdr = null;
                rc = sqlite3PagerGet(needSyncPgno, ref pPgHdr);
                if (rc != SQLITE.OK)
                {
                    if (needSyncPgno <= this.dbOrigSize)
                    {
                        Debug.Assert(this.pTmpSpace != null);
                        var pTemp = new uint[this.pTmpSpace.Length];
                        this.pInJournal.sqlite3BitvecClear(needSyncPgno, pTemp);
                    }
                    return rc;
                }
                pPgHdr.flags |= PgHdr.PGHDR.NEED_SYNC;
                PCache.sqlite3PcacheMakeDirty(pPgHdr);
                sqlite3PagerUnref(pPgHdr);
            }
            return SQLITE.OK;
        }
#endif
    }
}
