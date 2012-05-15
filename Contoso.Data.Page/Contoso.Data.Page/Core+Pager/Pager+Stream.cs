using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
using VFSLOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal SQLITE sqlite3PagerBegin(bool exFlag, int subjInMemory)
        {
            if (errCode != 0)
                return errCode;
            Debug.Assert(eState >= PAGER.READER && eState < PAGER.ERROR);
            subjInMemory = (byte)subjInMemory;
            var rc = SQLITE.OK;
            if (Check.ALWAYS(eState == PAGER.READER))
            {
                Debug.Assert(pInJournal == null);
                if (pagerUseWal())
                {
                    // If the pager is configured to use locking_mode=exclusive, and an exclusive lock on the database is not already held, obtain it now.
                    if (exclusiveMode && pWal.sqlite3WalExclusiveMode(-1))
                    {
                        rc = pagerLockDb(VFSLOCK.EXCLUSIVE);
                        if (rc != SQLITE.OK)
                            return rc;
                        pWal.sqlite3WalExclusiveMode(1);
                    }
                    // Grab the write lock on the log file. If successful, upgrade to PAGER_RESERVED state. Otherwise, return an error code to the caller.
                    // The busy-handler is not invoked if another connection already holds the write-lock. If possible, the upper layer will call it.
                    rc = pWal.sqlite3WalBeginWriteTransaction();
                }
                else
                {
                    // Obtain a RESERVED lock on the database file. If the exFlag parameter is true, then immediately upgrade this to an EXCLUSIVE lock. The
                    // busy-handler callback can be used when upgrading to the EXCLUSIVE lock, but not when obtaining the RESERVED lock.
                    rc = pagerLockDb(VFSLOCK.RESERVED);
                    if (rc == SQLITE.OK && exFlag)
                        rc = pager_wait_on_lock(VFSLOCK.EXCLUSIVE);
                }
                if (rc == SQLITE.OK)
                {
                    // Change to WRITER_LOCKED state.
                    // WAL mode sets Pager.eState to PAGER_WRITER_LOCKED or CACHEMOD when it has an open transaction, but never to DBMOD or FINISHED.
                    // This is because in those states the code to roll back savepoint transactions may copy data from the sub-journal into the database 
                    // file as well as into the page cache. Which would be incorrect in WAL mode.
                    eState = PAGER.WRITER_LOCKED;
                    dbHintSize = dbSize;
                    dbFileSize = dbSize;
                    dbOrigSize = dbSize;
                    journalOff = 0;
                }
                Debug.Assert(rc == SQLITE.OK || eState == PAGER.READER);
                Debug.Assert(rc != SQLITE.OK || eState == PAGER.WRITER_LOCKED);
                Debug.Assert(assert_pager_state());
            }
            PAGERTRACE("TRANSACTION {0}", PAGERID(this));
            return rc;
        }

        internal static SQLITE sqlite3PagerWrite(DbPage pDbPage)
        {
            var rc = SQLITE.OK;
            var pPg = pDbPage;
            var pPager = pPg.pPager;
            var nPagePerSector = (uint)(pPager.sectorSize / pPager.pageSize);
            Debug.Assert(pPager.eState >= PAGER.WRITER_LOCKED);
            Debug.Assert(pPager.eState != PAGER.ERROR);
            Debug.Assert(pPager.assert_pager_state());
            if (nPagePerSector > 1)
            {
                Pgno nPageCount = 0;     // Total number of pages in database file
                Pgno pg1;                // First page of the sector pPg is located on.
                Pgno nPage = 0;          // Number of pages starting at pg1 to journal
                bool needSync = false;   // True if any page has PGHDR_NEED_SYNC

                // Set the doNotSyncSpill flag to 1. This is because we cannot allow a journal header to be written between the pages journaled by
                // this function.
                Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
);
                Debug.Assert(pPager.doNotSyncSpill == 0);
                pPager.doNotSyncSpill++;
                // This trick assumes that both the page-size and sector-size are an integer power of 2. It sets variable pg1 to the identifier
                // of the first page of the sector pPg is located on.
                pg1 = (Pgno)((pPg.pgno - 1) & ~(nPagePerSector - 1)) + 1;
                nPageCount = pPager.dbSize;
                if (pPg.pgno > nPageCount)
                    nPage = (pPg.pgno - pg1) + 1;
                else if ((pg1 + nPagePerSector - 1) > nPageCount)
                    nPage = nPageCount + 1 - pg1;
                else
                    nPage = nPagePerSector;
                Debug.Assert(nPage > 0);
                Debug.Assert(pg1 <= pPg.pgno);
                Debug.Assert((pg1 + nPage) > pPg.pgno);
                for (var ii = 0; ii < nPage && rc == SQLITE.OK; ii++)
                {
                    var pg = (Pgno)(pg1 + ii);
                    var pPage = new PgHdr();
                    if (pg == pPg.pgno || pPager.pInJournal.sqlite3BitvecTest(pg) == 0)
                    {
                        if (pg != ((VirtualFile.PENDING_BYTE / (pPager.pageSize)) + 1))
                        {
                            rc = pPager.sqlite3PagerGet(pg, ref pPage);
                            if (rc == SQLITE.OK)
                            {
                                rc = pager_write(pPage);
                                if ((pPage.flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                                    needSync = true;
                                sqlite3PagerUnref(pPage);
                            }
                        }
                    }
                    else if ((pPage = pPager.pager_lookup(pg)) != null)
                    {
                        if ((pPage.flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                            needSync = true;
                        sqlite3PagerUnref(pPage);
                    }
                }
                // If the PGHDR_NEED_SYNC flag is set for any of the nPage pages starting at pg1, then it needs to be set for all of them. Because
                // writing to any of these nPage pages may damage the others, the journal file must contain sync()ed copies of all of them
                // before any of them can be written out to the database file.
                if (rc == SQLITE.OK && needSync)
                {
                    Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
);
                    for (var ii = 0; ii < nPage; ii++)
                    {
                        var pPage = pPager.pager_lookup((Pgno)(pg1 + ii));
                        if (pPage != null)
                        {
                            pPage.flags |= PgHdr.PGHDR.NEED_SYNC;
                            sqlite3PagerUnref(pPage);
                        }
                    }
                }
                Debug.Assert(pPager.doNotSyncSpill == 1);
                pPager.doNotSyncSpill--;
            }
            else
                rc = pager_write(pDbPage);
            return rc;
        }

        internal static void sqlite3PagerDontWrite(PgHdr pPg)
        {
            var pPager = pPg.pPager;
            if ((pPg.flags & PgHdr.PGHDR.DIRTY) != 0 && pPager.nSavepoint == 0)
            {
                PAGERTRACE("DONT_WRITE page {0} of {1}", pPg.pgno, PAGERID(pPager));
                SysEx.IOTRACE("CLEAN {0:x} {1}", pPager.GetHashCode(), pPg.pgno);
                pPg.flags |= PgHdr.PGHDR.DONT_WRITE;
                pager_set_pagehash(pPg);
            }
        }

        internal void sqlite3PagerTruncateImage(uint nPage)
        {
            Debug.Assert(dbSize >= nPage);
            Debug.Assert(eState >= PAGER.WRITER_CACHEMOD);
            dbSize = nPage;
            assertTruncateConstraint();
        }

        internal static SQLITE pagerStress(object p, PgHdr pPg)
        {
            var pPager = (Pager)p;
            var rc = SQLITE.OK;
            Debug.Assert(pPg.pPager == pPager);
            Debug.Assert((pPg.flags & PgHdr.PGHDR.DIRTY) != 0);
            // The doNotSyncSpill flag is set during times when doing a sync of journal (and adding a new header) is not allowed.  This occurs
            // during calls to sqlite3PagerWrite() while trying to journal multiple pages belonging to the same sector.
            // The doNotSpill flag inhibits all cache spilling regardless of whether or not a sync is required.  This is set during a rollback.
            // Spilling is also prohibited when in an error state since that could lead to database corruption.   In the current implementaton it 
            // is impossible for sqlite3PCacheFetch() to be called with createFlag==1 while in the error state, hence it is impossible for this routine to
            // be called in the error state.  Nevertheless, we include a NEVER() test for the error state as a safeguard against future changes.
            if (Check.NEVER(pPager.errCode != 0))
                return SQLITE.OK;
            if (pPager.doNotSpill != 0)
                return SQLITE.OK;
            if (pPager.doNotSyncSpill != 0 && (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                return SQLITE.OK;
            pPg.pDirty = null;
            if (pPager.pagerUseWal())
            {
                // Write a single frame for this page to the log.
                if (subjRequiresPage(pPg))
                    rc = subjournalPage(pPg);
                if (rc == SQLITE.OK)
                    rc = pPager.pagerWalFrames(pPg, 0, 0, 0);
            }
            else
            {
                // Sync the journal file if required. 
                if ((pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 || pPager.eState == PAGER.WRITER_CACHEMOD)
                    rc = pPager.syncJournal(1);
                // If the page number of this page is larger than the current size of the database image, it may need to be written to the sub-journal.
                // This is because the call to pager_write_pagelist() below will not actually write data to the file in this case.
                // Consider the following sequence of events:
                //   BEGIN;
                //     <journal page X>
                //     <modify page X>
                //     SAVEPOINT sp;
                //       <shrink database file to Y pages>
                //       pagerStress(page X)
                //     ROLLBACK TO sp;
                // If (X>Y), then when pagerStress is called page X will not be written out to the database file, but will be dropped from the cache. Then,
                // following the "ROLLBACK TO sp" statement, reading page X will read data from the database file. This will be the copy of page X as it
                // was when the transaction started, not as it was when "SAVEPOINT sp" was executed.
                // The solution is to write the current data for page X into the sub-journal file now (if it is not already there), so that it will
                // be restored to its current value when the "ROLLBACK TO sp" is executed.
                if (Check.NEVER(rc == SQLITE.OK && pPg.pgno > pPager.dbSize && subjRequiresPage(pPg)))
                    rc = subjournalPage(pPg);
                // Write the contents of the page out to the database file.
                if (rc == SQLITE.OK)
                {
                    Debug.Assert((pPg.flags & PgHdr.PGHDR.NEED_SYNC) == 0);
                    rc = pPager.pager_write_pagelist(pPg);
                }
            }
            // Mark the page as clean.
            if (rc == SQLITE.OK)
            {
                PAGERTRACE("STRESS {0} page {1}", PAGERID(pPager), pPg.pgno);
                PCache.sqlite3PcacheMakeClean(pPg);
            }
            return pPager.pager_error(rc);
        }
    }
}
