using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using System.Text;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal static PgHdr pager_lookup(Pager pPager, Pgno pgno)
        {
            // It is not possible for a call to PcacheFetch() with createFlag==0 to fail, since no attempt to allocate dynamic memory will be made.
            PgHdr p = null;
            sqlite3PcacheFetch(pPager.pPCache, pgno, 0, ref p);
            return p;
        }

        internal static void pager_reset(Pager pPager)
        {
            sqlite3BackupRestart(pPager.pBackup);
            sqlite3PcacheClear(pPager.pPCache);
        }

        internal static void releaseAllSavepoints(Pager pPager)
        {
            for (var ii = 0; ii < pPager.nSavepoint; ii++)
                sqlite3BitvecDestroy(ref pPager.aSavepoint[ii].pInSavepoint);
            if (!pPager.exclusiveMode || sqlite3IsMemJournal(pPager.sjfd))
                sqlite3OsClose(pPager.sjfd);
            pPager.aSavepoint = null;
            pPager.nSavepoint = 0;
            pPager.nSubRec = 0;
        }

        internal static SQLITE addToSavepointBitvecs(Pager pPager, Pgno pgno)
        {
            var rc = SQLITE.OK;
            for (var ii = 0; ii < pPager.nSavepoint; ii++)
            {
                var p = pPager.aSavepoint[ii];
                if (pgno <= p.nOrig)
                {
                    rc |= sqlite3BitvecSet(p.pInSavepoint, pgno);
                    Debug.Assert(rc == SQLITE.OK || rc == SQLITE.NOMEM);
                }
            }
            return rc;
        }

        internal static void pager_unlock(Pager pPager)
        {
            Debug.Assert(pPager.eState == PAGER.READER
                || pPager.eState == PAGER.OPEN
                || pPager.eState == PAGER.ERROR);
            sqlite3BitvecDestroy(ref pPager.pInJournal);
            pPager.pInJournal = null;
            releaseAllSavepoints(pPager);
            if (pagerUseWal(pPager))
            {
                Debug.Assert(!pPager.jfd.isOpen);
                sqlite3WalEndReadTransaction(pPager.pWal);
                pPager.eState = PAGER.OPEN;
            }
            else if (!pPager.exclusiveMode)
            {
                var iDc = (pPager.fd.isOpen ? sqlite3OsDeviceCharacteristics(pPager.fd) : 0);
                // If the operating system support deletion of open files, then close the journal file when dropping the database lock.  Otherwise
                // another connection with journal_mode=delete might delete the file out from under us.
                Debug.Assert(((int)JOURNALMODE.MEMORY & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.OFF & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.WAL & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.DELETE & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert(((int)JOURNALMODE.PERSIST & 5) == 1);
                if (0 == (iDc & SQLITE_IOCAP_UNDELETABLE_WHEN_OPEN) || 1 != ((int)pPager.journalMode & 5))
                    sqlite3OsClose(pPager.jfd);
                // If the pager is in the ERROR state and the call to unlock the database file fails, set the current lock to UNKNOWN_LOCK. See the comment
                // above the #define for UNKNOWN_LOCK for an explanation of why this is necessary.
                var rc = pagerUnlockDb(pPager, LOCK.NO);
                if (rc != SQLITE.OK && pPager.eState == PAGER.ERROR)
                    pPager.eLock = LOCK.UNKNOWN;
                // The pager state may be changed from PAGER_ERROR to PAGER_OPEN here without clearing the error code. This is intentional - the error
                // code is cleared and the cache reset in the block below.
                Debug.Assert(pPager.errCode != 0 || pPager.eState != PAGER.ERROR);
                pPager.changeCountDone = false;
                pPager.eState = PAGER.OPEN;
            }
            // If Pager.errCode is set, the contents of the pager cache cannot be trusted. Now that there are no outstanding references to the pager,
            // it can safely move back to PAGER_OPEN state. This happens in both normal and exclusive-locking mode.
            if (pPager.errCode != 0)
            {
                Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
);
                pager_reset(pPager);
                pPager.changeCountDone = pPager.tempFile;
                pPager.eState = PAGER.OPEN;
                pPager.errCode = SQLITE.OK;
            }
            pPager.journalOff = 0;
            pPager.journalHdr = 0;
            pPager.setMaster = 0;
        }

        internal static SQLITE pager_error(Pager pPager, SQLITE rc)
        {
            var rc2 = (SQLITE)((int)rc & 0xff);
            Debug.Assert(rc == SQLITE.OK ||
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
 0 == pPager.memDb
#endif
);
            Debug.Assert(pPager.errCode == SQLITE.FULL || pPager.errCode == SQLITE.OK || ((int)pPager.errCode & 0xff) == (int)SQLITE.IOERR);
            if (rc2 == SQLITE.FULL || rc2 == SQLITE.IOERR)
            {
                pPager.errCode = rc;
                pPager.eState = PAGER.ERROR;
            }
            return rc;
        }

        internal static SQLITE pager_end_transaction(Pager pPager, int hasMaster)
        {
            // Do nothing if the pager does not have an open write transaction or at least a RESERVED lock. This function may be called when there
            // is no write-transaction active but a RESERVED or greater lock is held under two circumstances:
            //   1. After a successful hot-journal rollback, it is called with eState==PAGER_NONE and eLock==EXCLUSIVE_LOCK.
            //   2. If a connection with locking_mode=exclusive holding an EXCLUSIVE lock switches back to locking_mode=normal and then executes a
            //      read-transaction, this function is called with eState==PAGER_READER and eLock==EXCLUSIVE_LOCK when the read-transaction is closed.
            Debug.Assert(assert_pager_state(pPager));
            Debug.Assert(pPager.eState != PAGER.ERROR);
            if (pPager.eState < PAGER.WRITER_LOCKED && pPager.eLock < LOCK.RESERVED)
                return SQLITE.OK;
            releaseAllSavepoints(pPager);
            Debug.Assert(pPager.jfd.isOpen || pPager.pInJournal == null);
            var rc = SQLITE.OK; // Error code from journal finalization operation
            if (pPager.jfd.isOpen)
            {
                Debug.Assert(!pagerUseWal(pPager));
                // Finalize the journal file.
                if (sqlite3IsMemJournal(pPager.jfd))
                {
                    Debug.Assert(pPager.journalMode == JOURNALMODE.MEMORY);
                    sqlite3OsClose(pPager.jfd);
                }
                else if (pPager.journalMode == JOURNALMODE.TRUNCATE)
                {
                    rc = (pPager.journalOff == 0 ? SQLITE.OK : sqlite3OsTruncate(pPager.jfd, 0));
                    pPager.journalOff = 0;
                }
                else if (pPager.journalMode == JOURNALMODE.PERSIST || (pPager.exclusiveMode && pPager.journalMode != JOURNALMODE.WAL))
                {
                    rc = zeroJournalHdr(pPager, hasMaster);
                    pPager.journalOff = 0;
                }
                else
                {
                    // This branch may be executed with Pager.journalMode==MEMORY if a hot-journal was just rolled back. In this case the journal
                    // file should be closed and deleted. If this connection writes to the database file, it will do so using an in-memory journal. 
                    Debug.Assert(pPager.journalMode == JOURNALMODE.DELETE || pPager.journalMode == JOURNALMODE.MEMORY || pPager.journalMode == JOURNALMODE.WAL);
                    sqlite3OsClose(pPager.jfd);
                    if (!pPager.tempFile)
                        rc = sqlite3OsDelete(pPager.pVfs, pPager.zJournal, 0);
                }
            }
#if SQLITE_CHECK_PAGES
            sqlite3PcacheIterateDirty(pPager.pPCache, pager_set_pagehash);
            if (pPager.dbSize == 0 && sqlite3PcacheRefCount(pPager.pPCache) > 0)
            {
                var p = pager_lookup(pPager, 1);
                if (p != null)
                {
                    p.pageHash = null;
                    sqlite3PagerUnref(p);
                }
            }
#endif
            sqlite3BitvecDestroy(ref pPager.pInJournal);
            pPager.pInJournal = null;
            pPager.nRec = 0;
            sqlite3PcacheCleanAll(pPager.pPCache);
            sqlite3PcacheTruncate(pPager.pPCache, pPager.dbSize);
            var rc2 = SQLITE.OK;    // Error code from db file unlock operation
            if (pagerUseWal(pPager))
            {
                // Drop the WAL write-lock, if any. Also, if the connection was in locking_mode=exclusive mode but is no longer, drop the EXCLUSIVE 
                // lock held on the database file.
                rc2 = sqlite3WalEndWriteTransaction(pPager.pWal);
                Debug.Assert(rc2 == SQLITE.OK);
            }
            if (!pPager.exclusiveMode && (!pagerUseWal(pPager) || sqlite3WalExclusiveMode(pPager.pWal, 0)))
            {
                rc2 = pagerUnlockDb(pPager, LOCK.SHARED);
                pPager.changeCountDone = false;
            }
            pPager.eState = PAGER.READER;
            pPager.setMaster = 0;
            return (rc == SQLITE.OK ? rc2 : rc);
        }

        internal static void pagerUnlockAndRollback(Pager pPager)
        {
            if (pPager.eState != PAGER.ERROR && pPager.eState != PAGER.OPEN)
            {
                Debug.Assert(assert_pager_state(pPager));
                if (pPager.eState >= PAGER.WRITER_LOCKED)
                {
                    sqlite3BeginBenignMalloc();
                    sqlite3PagerRollback(pPager);
                    sqlite3EndBenignMalloc();
                }
                else if (!pPager.exclusiveMode)
                {
                    Debug.Assert(pPager.eState == PAGER.READER);
                    pager_end_transaction(pPager, 0);
                }
            }
            pager_unlock(pPager);
        }

        internal static uint pager_cksum(Pager pPager, byte[] aData)
        {
            var cksum = pPager.cksumInit;
            var i = pPager.pageSize - 200;
            while (i > 0)
            {
                cksum += aData[i];
                i -= 200;
            }
            return cksum;
        }

#if SQLITE_HAS_CODEC
        static void pagerReportSize(Pager pPager)
        {
            if (pPager.xCodecSizeChng != null)
            {
                pPager.xCodecSizeChng(pPager.pCodec, pPager.pageSize,
                pPager.nReserve);
            }
        }
#else
        internal static void pagerReportSize(Pager X) { }
#endif

        internal static SQLITE pager_playback_one_page(Pager pPager, ref long pOffset, Bitvec pDone, int isMainJrnl, int isSavepnt)
        {
            Debug.Assert((isMainJrnl & ~1) == 0);       // isMainJrnl is 0 or 1
            Debug.Assert((isSavepnt & ~1) == 0);        // isSavepnt is 0 or 1
            Debug.Assert(isMainJrnl != 0 || pDone != null);     // pDone always used on sub-journals
            Debug.Assert(isSavepnt != 0 || pDone == null);      // pDone never used on non-savepoint
            var aData = pPager.pTmpSpace;
            Debug.Assert(aData != null);         // Temp storage must have already been allocated
            Debug.Assert(!pagerUseWal(pPager) || (0 == isMainJrnl && isSavepnt != 0));
            // Either the state is greater than PAGER_WRITER_CACHEMOD (a transaction or savepoint rollback done at the request of the caller) or this is
            // a hot-journal rollback. If it is a hot-journal rollback, the pager is in state OPEN and holds an EXCLUSIVE lock. Hot-journal rollback
            // only reads from the main journal, not the sub-journal.
            Debug.Assert(pPager.eState >= PAGER.WRITER_CACHEMOD || (pPager.eState == PAGER.OPEN && pPager.eLock == LOCK.EXCLUSIVE));
            Debug.Assert(pPager.eState >= PAGER.WRITER_CACHEMOD || isMainJrnl != 0);
            // Read the page number and page data from the journal or sub-journal file. Return an error code to the caller if an IO error occurs.
            var jfd = (isMainJrnl != 0 ? pPager.jfd : pPager.sjfd); // The file descriptor for the journal file
            Pgno pgno = 0;  // The page number of a page in journal
            var rc = read32bits(jfd, pOffset, ref pgno);
            if (rc != SQLITE.OK)
                return rc;
            rc = sqlite3OsRead(jfd, aData, pPager.pageSize, pOffset + 4);
            if (rc != SQLITE.OK)
                return rc;
            pOffset += pPager.pageSize + 4 + isMainJrnl * 4;
            // Sanity checking on the page.  This is more important that I originally thought.  If a power failure occurs while the journal is being written,
            // it could cause invalid data to be written into the journal.  We need to detect this invalid data (with high probability) and ignore it.
            if (pgno == 0 || pgno == PAGER_MJ_PGNO(pPager))
            {
                Debug.Assert(0 == isSavepnt);
                return SQLITE.DONE;
            }
            if (pgno > pPager.dbSize || sqlite3BitvecTest(pDone, pgno) != 0)
                return SQLITE.OK;
            uint cksum = 0;         // Checksum used for sanity checking
            if (isMainJrnl != 0)
            {
                rc = read32bits(jfd, (pOffset) - 4, ref cksum);
                if (rc != SQLITE.OK)
                    return rc;
                if (0 == isSavepnt && pager_cksum(pPager, aData) != cksum)
                    return SQLITE.DONE;
            }
            // If this page has already been played by before during the current rollback, then don't bother to play it back again.
            if (pDone != null && (rc = sqlite3BitvecSet(pDone, pgno)) != SQLITE.OK)
                return rc;
            // When playing back page 1, restore the nReserve setting
            if (pgno == 1 && pPager.nReserve != aData[20])
            {
                pPager.nReserve = (aData)[20];
                pagerReportSize(pPager);
            }
            var pPg = (pagerUseWal(pPager) ? null : pager_lookup(pPager, pgno)); // An existing page in the cache
            Debug.Assert(pPg != null ||
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
 pPager.memDb == 0
#endif
);
            Debug.Assert(pPager.eState != PAGER.OPEN || pPg == null);
            PAGERTRACE("PLAYBACK %d page %d hash(%08x) %s\n",
                PAGERID(pPager),
                pgno,
                pager_datahash(pPager.pageSize, aData),
                isMainJrnl != 0 ? "main-journal" : "sub-journal");
            bool isSynced; // True if journal page is synced
            if (isMainJrnl != 0)
                isSynced = pPager.noSync || (pOffset <= pPager.journalHdr);
            else
                isSynced = (pPg == null || 0 == (pPg.flags & PgHdr.PGHDR.NEED_SYNC));
            if (pPager.fd.isOpen && (pPager.eState >= PAGER.WRITER_DBMOD || pPager.eState == PAGER.OPEN) && isSynced)
            {
                long ofst = (pgno - 1) * pPager.pageSize;
                Debug.Assert(!pagerUseWal(pPager));
                rc = sqlite3OsWrite(pPager.fd, aData, pPager.pageSize, ofst);
                if (pgno > pPager.dbFileSize)
                    pPager.dbFileSize = pgno;
                if (pPager.pBackup != null)
                {
                    if (CODEC1(pPager, aData, pgno, SQLITE_DECRYPT))
                        rc = SQLITE.NOMEM;
                    sqlite3BackupUpdate(pPager.pBackup, pgno, aData);
                    if (CODEC2(pPager, aData, pgno, SQLITE_ENCRYPT_READ_CTX, ref aData))
                        rc = SQLITE.NOMEM;
                }
            }
            else if (0 == isMainJrnl && pPg == null)
            {
                // If this is a rollback of a savepoint and data was not written to the database and the page is not in-memory, there is a potential
                // problem. When the page is next fetched by the b-tree layer, it will be read from the database file, which may or may not be
                // current.
                // There are a couple of different ways this can happen. All are quite obscure. When running in synchronous mode, this can only happen
                // if the page is on the free-list at the start of the transaction, then populated, then moved using sqlite3PagerMovepage().
                // The solution is to add an in-memory page to the cache containing the data just read from the sub-journal. Mark the page as dirty
                // and if the pager requires a journal-sync, then mark the page as requiring a journal-sync before it is written.
                Debug.Assert(isSavepnt != 0);
                Debug.Assert(pPager.doNotSpill == 0);
                pPager.doNotSpill++;
                rc = sqlite3PagerAcquire(pPager, pgno, ref pPg, 1);
                Debug.Assert(pPager.doNotSpill == 1);
                pPager.doNotSpill--;
                if (rc != SQLITE.OK)
                    return rc;
                pPg.flags &= ~PgHdr.PGHDR.NEED_READ;
                sqlite3PcacheMakeDirty(pPg);
            }
            if (pPg != null)
            {
                // No page should ever be explicitly rolled back that is in use, except for page 1 which is held in use in order to keep the lock on the
                // database active. However such a page may be rolled back as a result of an internal error resulting in an automatic call to
                // sqlite3PagerRollback().
                var pData = pPg.pData;
                Buffer.BlockCopy(aData, 0, pData, 0, pPager.pageSize);
                pPager.xReiniter(pPg);
                if (isMainJrnl != 0 && (0 == isSavepnt || pOffset <= pPager.journalHdr))
                {
                    // If the contents of this page were just restored from the main journal file, then its content must be as they were when the
                    // transaction was first opened. In this case we can mark the page as clean, since there will be no need to write it out to the
                    // database.
                    // There is one exception to this rule. If the page is being rolled back as part of a savepoint (or statement) rollback from an
                    // unsynced portion of the main journal file, then it is not safe to mark the page as clean. This is because marking the page as
                    // clean will clear the PGHDR_NEED_SYNC flag. Since the page is already in the journal file (recorded in Pager.pInJournal) and
                    // the PGHDR_NEED_SYNC flag is cleared, if the page is written to again within this transaction, it will be marked as dirty but
                    // the PGHDR_NEED_SYNC flag will not be set. It could then potentially be written out into the database file before its journal file
                    // segment is synced. If a crash occurs during or following this, database corruption may ensue.
                    Debug.Assert(!pagerUseWal(pPager));
                    sqlite3PcacheMakeClean(pPg);
                }
                pager_set_pagehash(pPg);
                // If this was page 1, then restore the value of Pager.dbFileVers. Do this before any decoding.
                if (pgno == 1)
                    Buffer.BlockCopy(pData, 24, pPager.dbFileVers, 0, pPager.dbFileVers.Length);
                // Decode the page just read from disk
                if (CODEC1(pPager, pData, pPg.pgno, SQLITE_DECRYPT))
                    rc = SQLITE.NOMEM;
                sqlite3PcacheRelease(pPg);
            }
            return rc;
        }

        internal static int pager_delmaster(Pager pPager, string zMaster)
        {
            var pVfs = pPager.pVfs;
            //string zMasterJournal = null; /* Contents of master journal file */
            long nMasterJournal;           /* Size of master journal file */
            string zJournal;              /* Pointer to one journal within MJ file */
            string zMasterPtr;            /* Space to hold MJ filename from a journal file */
            int nMasterPtr;               /* Amount of space allocated to zMasterPtr[] */

            // Allocate space for both the pJournal and pMaster file descriptors. If successful, open the master journal file for reading.
            var pMaster = new sqlite3_file();// Malloc'd master-journal file descriptor
            var pJournal = new sqlite3_file();// Malloc'd child-journal file descriptor
            int iDummy = 0;
            var rc = sqlite3OsOpen(pVfs, zMaster, pMaster, SQLITE_OPEN_READONLY | SQLITE_OPEN_MASTER_JOURNAL, ref iDummy);
            if (rc != SQLITE.OK)
                goto delmaster_out;
            Debugger.Break();
            //TODO --
            // Load the entire master journal file into space obtained from sqlite3_malloc() and pointed to by zMasterJournal.   Also obtain
            // sufficient space (in zMasterPtr) to hold the names of master journal files extracted from regular rollback-journals.
            //rc = sqlite3OsFileSize(pMaster, &nMasterJournal);
            //if (rc != SQLITE_OK) goto delmaster_out;
            //nMasterPtr = pVfs.mxPathname + 1;
            //  zMasterJournal = sqlite3Malloc((int)nMasterJournal + nMasterPtr + 1);
            //  if ( !zMasterJournal )
            //  {
            //    rc = SQLITE_NOMEM;
            //    goto delmaster_out;
            //  }
            //  zMasterPtr = &zMasterJournal[nMasterJournal+1];
            //  rc = sqlite3OsRead( pMaster, zMasterJournal, (int)nMasterJournal, 0 );
            //  if ( rc != SQLITE_OK ) goto delmaster_out;
            //  zMasterJournal[nMasterJournal] = 0;
            //  zJournal = zMasterJournal;
            //  while ( ( zJournal - zMasterJournal ) < nMasterJournal )
            //  {
            //    int exists;
            //    rc = sqlite3OsAccess( pVfs, zJournal, SQLITE_ACCESS_EXISTS, &exists );
            //    if ( rc != SQLITE_OK )
            //      goto delmaster_out;
            //    if ( exists )
            //    {
            //      // One of the journals pointed to by the master journal exists. Open it and check if it points at the master journal. If
            //      // so, return without deleting the master journal file.
            //      int c;
            //      int flags = ( SQLITE_OPEN_READONLY | SQLITE_OPEN_MAIN_JOURNAL );
            //      rc = sqlite3OsOpen( pVfs, zJournal, pJournal, flags, 0 );
            //      if ( rc != SQLITE_OK )
            //        goto delmaster_out;
            //      rc = readMasterJournal( pJournal, zMasterPtr, nMasterPtr );
            //      sqlite3OsClose( pJournal );
            //      if ( rc != SQLITE_OK )
            //        goto delmaster_out;
            //      c = zMasterPtr[0] != 0 && strcmp( zMasterPtr, zMaster ) == 0;
            //      if ( c )
            //        // We have a match. Do not delete the master journal file.
            //        goto delmaster_out;
            //    }
            //    zJournal += ( sqlite3Strlen30( zJournal ) + 1 );
            //   }
            //
            //sqlite3OsClose(pMaster);
            //rc = sqlite3OsDelete( pVfs, zMaster, 0 );
            goto delmaster_out;
        delmaster_out:
            if (pMaster != null)
            {
                sqlite3OsClose(pMaster);
                Debug.Assert(!pJournal.isOpen);
            }
            return rc;
        }

        internal static SQLITE pager_truncate(Pager pPager, Pgno nPage)
        {
            Debug.Assert(pPager.eState != PAGER.ERROR);
            Debug.Assert(pPager.eState != PAGER.READER);
            var rc = SQLITE.OK;
            if (pPager.fd.isOpen && (pPager.eState >= PAGER.WRITER_DBMOD || pPager.eState == PAGER.OPEN))
            {
                var szPage = pPager.pageSize;
                Debug.Assert(pPager.eLock == LOCK.EXCLUSIVE);
                // TODO: Is it safe to use Pager.dbFileSize here?
                long currentSize = 0;
                rc = sqlite3OsFileSize(pPager.fd, ref currentSize);
                var newSize = szPage * nPage;
                if (rc == SQLITE.OK && currentSize != newSize)
                {
                    if (currentSize > newSize)
                        rc = sqlite3OsTruncate(pPager.fd, newSize);
                    else
                    {
                        var pTmp = pPager.pTmpSpace;
                        Array.Clear(pTmp, 0, szPage);
                        rc = sqlite3OsWrite(pPager.fd, pTmp, szPage, newSize - szPage);
                    }
                    if (rc == SQLITE.OK)
                        pPager.dbSize = nPage;
                }
            }
            return rc;
        }

        internal static void setSectorSize(Pager pPager)
        {
            Debug.Assert(pPager.fd.isOpen || pPager.tempFile);
            if (!pPager.tempFile)
            {
                // Sector size doesn't matter for temporary files. Also, the file may not have been opened yet, in which case the OsSectorSize()
                // call will segfault.
                pPager.sectorSize = (Pgno)sqlite3OsSectorSize(pPager.fd);
            }
            if (pPager.sectorSize < 32)
            {
                Debug.Assert(MAX_SECTOR_SIZE >= 512);
                pPager.sectorSize = 512;
            }
            if (pPager.sectorSize > MAX_SECTOR_SIZE)
                pPager.sectorSize = MAX_SECTOR_SIZE;
        }

        internal static int pager_playback(Pager pPager, int isHot)
        {
            var pVfs = pPager.pVfs;
            // Figure out how many records are in the journal.  Abort early if the journal is empty.
            Debug.Assert(pPager.jfd.isOpen);
            long szJ = 0;            // Size of the journal file in bytes
            var rc = sqlite3OsFileSize(pPager.jfd, ref szJ);
            if (rc != SQLITE.OK)
                goto end_playback;
            // Read the master journal name from the journal, if it is present. If a master journal file name is specified, but the file is not
            // present on disk, then the journal is not hot and does not need to be played back.
            //
            // TODO: Technically the following is an error because it assumes that buffer Pager.pTmpSpace is (mxPathname+1) bytes or larger. i.e. that
            // (pPager.pageSize >= pPager.pVfs.mxPathname+1). Using os_unix.c, mxPathname is 512, which is the same as the minimum allowable value
            // for pageSize.
            var zMaster = new byte[pPager.pVfs.mxPathname + 1]; // Name of master journal file if any
            rc = readMasterJournal(pPager.jfd, zMaster, (uint)pPager.pVfs.mxPathname + 1);
            var res = 1;             // Value returned by sqlite3OsAccess()
            if (rc == SQLITE.OK && zMaster[0] != 0)
                rc = sqlite3OsAccess(pVfs, Encoding.UTF8.GetString(zMaster, 0, zMaster.Length), SQLITE_ACCESS_EXISTS, ref res);
            zMaster = null;
            if (rc != SQLITE.OK || res == 0)
                goto end_playback;
            pPager.journalOff = 0;
            // This loop terminates either when a readJournalHdr() or pager_playback_one_page() call returns SQLITE_DONE or an IO error occurs.
            var needPagerReset = isHot; // True to reset page prior to first page rollback
            uint nRec = 0;            // Number of Records in the journal
            while (true)
            {
                // Read the next journal header from the journal file.  If there are not enough bytes left in the journal file for a complete header, or
                // it is corrupted, then a process must have failed while writing it. This indicates nothing more needs to be rolled back.
                rc = readJournalHdr(pPager, isHot, szJ, out nRec, out mxPg);
                if (rc != SQLITE.OK)
                {
                    if (rc == SQLITE.DONE)
                        rc = SQLITE.OK;
                    goto end_playback;
                }
                // If nRec is 0xffffffff, then this journal was created by a process working in no-sync mode. This means that the rest of the journal
                // file consists of pages, there are no more journal headers. Compute the value of nRec based on this assumption.
                if (nRec == 0xffffffff)
                {
                    Debug.Assert(pPager.journalOff == JOURNAL_HDR_SZ(pPager));
                    nRec = (uint)((szJ - JOURNAL_HDR_SZ(pPager)) / JOURNAL_PG_SZ(pPager));
                }
                // If nRec is 0 and this rollback is of a transaction created by this process and if this is the final header in the journal, then it means
                // that this part of the journal was being filled but has not yet been synced to disk.  Compute the number of pages based on the remaining
                // size of the file.
                // The third term of the test was added to fix ticket #2565. When rolling back a hot journal, nRec==0 always means that the next
                // chunk of the journal contains zero pages to be rolled back.  But when doing a ROLLBACK and the nRec==0 chunk is the last chunk in
                // the journal, it means that the journal might contain additional pages that need to be rolled back and that the number of pages
                // should be computed based on the journal file size.
                if (nRec == 0 && 0 == isHot && pPager.journalHdr + JOURNAL_HDR_SZ(pPager) == pPager.journalOff)
                    nRec = (uint)((szJ - pPager.journalOff) / JOURNAL_PG_SZ(pPager));
                // If this is the first header read from the journal, truncate the database file back to its original size.
                if (pPager.journalOff == JOURNAL_HDR_SZ(pPager))
                {
                    Pgno mxPg = 0; // Size of the original file in pages
                    rc = pager_truncate(pPager, mxPg);
                    if (rc != SQLITE.OK)
                        goto end_playback;
                    pPager.dbSize = mxPg;
                }
                // Copy original pages out of the journal and back into the database file and/or page cache.
                for (var u = 0; u < nRec; u++)
                {
                    if (needPagerReset != 0)
                    {
                        pager_reset(pPager);
                        needPagerReset = 0;
                    }
                    rc = pager_playback_one_page(pPager, ref pPager.journalOff, null, 1, 0);
                    if (rc != SQLITE.OK)
                    {
                        if (rc == SQLITE.DONE)
                        {
                            rc = SQLITE.OK;
                            pPager.journalOff = szJ;
                            break;
                        }
                        else if (rc == SQLITE_IOERR_SHORT_READ)
                        {
                            // If the journal has been truncated, simply stop reading and processing the journal. This might happen if the journal was
                            // not completely written and synced prior to a crash.  In that case, the database should have never been written in the
                            // first place so it is OK to simply abandon the rollback.
                            rc = SQLITE.OK;
                            goto end_playback;
                        }
                        else
                            // If we are unable to rollback, quit and return the error code.  This will cause the pager to enter the error state
                            // so that no further harm will be done.  Perhaps the next process to come along will be able to rollback the database.
                            goto end_playback;
                    }
                }
            }
        end_playback:
            // Following a rollback, the database file should be back in its original state prior to the start of the transaction, so invoke the
            // SQLITE_FCNTL_DB_UNCHANGED file-control method to disable the assertion that the transaction counter was modified.
            long iDummy = 0;
            Debug.Assert(!pPager.fd.isOpen || sqlite3OsFileControl(pPager.fd, SQLITE_FCNTL_DB_UNCHANGED, ref iDummy) >= SQLITE.OK);

            // If this playback is happening automatically as a result of an IO or malloc error that occurred after the change-counter was updated but
            // before the transaction was committed, then the change-counter modification may just have been reverted. If this happens in exclusive
            // mode, then subsequent transactions performed by the connection will not update the change-counter at all. This may lead to cache inconsistency
            // problems for other processes at some point in the future. So, just in case this has happened, clear the changeCountDone flag now.
            pPager.changeCountDone = pPager.tempFile;
            if (rc == SQLITE.OK)
            {
                zMaster = new byte[pPager.pVfs.mxPathname + 1];
                rc = readMasterJournal(pPager.jfd, zMaster, (uint)pPager.pVfs.mxPathname + 1);
            }
            if (rc == SQLITE.OK && (pPager.eState >= PAGER.WRITER_DBMOD || pPager.eState == PAGER.OPEN))
                rc = sqlite3PagerSync(pPager);
            if (rc == SQLITE.OK)
                rc = pager_end_transaction(pPager, zMaster[0] != '\0' ? 1 : 0);
            if (rc == SQLITE.OK && zMaster[0] != '\0' && res != 0)
                // If there was a master journal and this routine will return success, see if it is possible to delete the master journal.
                rc = pager_delmaster(pPager, Encoding.UTF8.GetString(zMaster, 0, zMaster.Length));
            // The Pager.sectorSize variable may have been updated while rolling back a journal created by a process with a different sector size
            // value. Reset it to the correct value for this process.
            setSectorSize(pPager);
            return rc;
        }




        internal static SQLITE readDbPage(PgHdr pPg)
        {
            var pPager = pPg.pPager;    // Pager object associated with page pPg
            var pgno = pPg.pgno;        // Page number to read
            var rc = SQLITE.OK;         // Return code
            var isInWal = 0;            // True if page is in log file
            var pgsz = pPager.pageSize; // Number of bytes to read
            Debug.Assert(pPager.eState >= PAGER.READER &&
#if SQLITE_OMIT_MEMORYDB
0 == MEMDB 
#else
 0 == pPager.memDb
#endif
);
            Debug.Assert(pPager.fd.isOpen);
            if (Check.NEVER(!pPager.fd.isOpen))
            {
                Debug.Assert(pPager.tempFile);
                Array.Clear(pPg.pData, 0, pPager.pageSize);
                return SQLITE.OK;
            }
            if (pagerUseWal(pPager))
                // Try to pull the page from the write-ahead log.
                rc = sqlite3WalRead(pPager.pWal, pgno, ref isInWal, pgsz, pPg.pData);
            if (rc == SQLITE.OK && 0 == isInWal)
            {
                long iOffset = (pgno - 1) * (long)pPager.pageSize;
                rc = sqlite3OsRead(pPager.fd, pPg.pData, pgsz, iOffset);
                if (rc == SQLITE.IOERR_SHORT_READ)
                    rc = SQLITE_OK;
            }
            if (pgno == 1)
            {
                if (rc != 0)
                    // If the read is unsuccessful, set the dbFileVers[] to something that will never be a valid file version.  dbFileVers[] is a copy
                    // of bytes 24..39 of the database.  Bytes 28..31 should always be zero or the size of the database in page. Bytes 32..35 and 35..39
                    // should be page numbers which are never 0xffffffff.  So filling pPager.dbFileVers[] with all 0xff bytes should suffice.
                    //
                    // For an encrypted database, the situation is more complex:  bytes 24..39 of the database are white noise.  But the probability of
                    // white noising equaling 16 bytes of 0xff is vanishingly small so we should still be ok.
                    for (int i = 0; i < pPager.dbFileVers.Length; pPager.dbFileVers[i++] = 0xff) ; // memset(pPager.dbFileVers, 0xff, sizeof(pPager.dbFileVers));
                else
                    Buffer.BlockCopy(pPg.pData, 24, pPager.dbFileVers, 0, pPager.dbFileVers.Length);
            }
            if (CODEC1(pPager, pPg.pData, pgno, SQLITE_DECRYPT))
                rc = SQLITE.NOMEM;
            IOTRACE("PGIN %p %d\n", pPager, pgno);
            PAGERTRACE("FETCH %d page %d hash(%08x)\n",
            PAGERID(pPager), pgno, pager_pagehash(pPg));
            return rc;
        }

        internal static void pager_write_changecounter(PgHdr pPg)
        {
            // Increment the value just read and write it back to byte 24.
            uint change_counter = sqlite3Get4byte(pPg.pPager.dbFileVers, 0) + 1;
            put32bits(pPg.pData, 24, change_counter);
            // Also store the SQLite version number in bytes 96..99 and in bytes 92..95 store the change counter for which the version number
            // is valid.
            put32bits(pPg.pData, 92, change_counter);
            put32bits(pPg.pData, 96, SQLITE_VERSION_NUMBER);
        }

        internal static SQLITE pagerPagecount(Pager pPager, ref Pgno pnPage)
        {
            // Query the WAL sub-system for the database size. The WalDbsize() function returns zero if the WAL is not open (i.e. Pager.pWal==0), or
            // if the database size is not available. The database size is not available from the WAL sub-system if the log file is empty or
            // contains no valid committed transactions.
            Debug.Assert(pPager.eState == PAGER.OPEN);
            Debug.Assert(pPager.eLock >= LOCK.SHARED || pPager.noReadlock != 0);
            var nPage = sqlite3WalDbsize(pPager.pWal);
            // If the database size was not available from the WAL sub-system, determine it based on the size of the database file. If the size
            // of the database file is not an integer multiple of the page-size, round down to the nearest page. Except, any file larger than 0
            // bytes in size is considered to contain at least one page.
            if (nPage == 0)
            {
                var n = 0L; // Size of db file in bytes
                Debug.Assert(pPager.fd.isOpen || pPager.tempFile);
                if (pPager.fd.isOpen)
                {
                    var rc = sqlite3OsFileSize(pPager.fd, ref n);
                    if (rc != SQLITE.OK)
                        return rc;
                }
                nPage = (Pgno)(n / pPager.pageSize);
                if (nPage == 0 && n > 0)
                    nPage = 1;
            }
            // If the current number of pages in the file is greater than the configured maximum pager number, increase the allowed limit so
            // that the file can be read.
            if (nPage > pPager.mxPgno)
                pPager.mxPgno = (Pgno)nPage;
            pnPage = nPage;
            return SQLITE.OK;
        }

        internal static SQLITE pagerPlaybackSavepoint(Pager pPager, PagerSavepoint pSavepoint)
        {
            Debug.Assert(pPager.eState != PAGER.ERROR);
            Debug.Assert(pPager.eState >= PAGER.WRITER_LOCKED);
            // Allocate a bitvec to use to store the set of pages rolled back
            Bitvec pDone = null;     // Bitvec to ensure pages played back only once
            if (pSavepoint != null)
                pDone = sqlite3BitvecCreate(pSavepoint.nOrig);
            // Set the database size back to the value it was before the savepoint being reverted was opened.
            pPager.dbSize = pSavepoint != null ? pSavepoint.nOrig : pPager.dbOrigSize;
            pPager.changeCountDone = pPager.tempFile;
            if (!pSavepoint && pagerUseWal(pPager))
                return pagerRollbackWal(pPager);
            // Use pPager.journalOff as the effective size of the main rollback journal.  The actual file might be larger than this in
            // PAGER_JOURNALMODE_TRUNCATE or PAGER_JOURNALMODE_PERSIST.  But anything past pPager.journalOff is off-limits to us.
            var szJ = pPager.journalOff; // Effective size of the main journal
            Debug.Assert(!pagerUseWal(pPager) || szJ == 0);
            // Begin by rolling back records from the main journal starting at PagerSavepoint.iOffset and continuing to the next journal header.
            // There might be records in the main journal that have a page number greater than the current database size (pPager.dbSize) but those
            // will be skipped automatically.  Pages are added to pDone as they are played back.
            long iHdrOff;             // End of first segment of main-journal records
            var rc = SQLITE.OK;      // Return code
            if (pSavepoint != null && !pagerUseWal(pPager))
            {
                iHdrOff = pSavepoint.iHdrOffset != 0 ? pSavepoint.iHdrOffset : szJ;
                pPager.journalOff = pSavepoint.iOffset;
                while (rc == SQLITE.OK && pPager.journalOff < iHdrOff)
                    rc = pager_playback_one_page(pPager, ref pPager.journalOff, pDone, 1, 1);
                Debug.Assert(rc != SQLITE.DONE);
            }
            else
                pPager.journalOff = 0;
            // Continue rolling back records out of the main journal starting at the first journal header seen and continuing until the effective end
            // of the main journal file.  Continue to skip out-of-range pages and continue adding pages rolled back to pDone.
            while (rc == SQLITE.OK && pPager.journalOff < szJ)
            {
                uint dummy;
                rc = readJournalHdr(pPager, 0, (int)szJ, out nJRec, out dummy);
                Debug.Assert(rc != SQLITE.DONE);
                // The "pPager.journalHdr+JOURNAL_HDR_SZ(pPager)==pPager.journalOff" test is related to ticket #2565.  See the discussion in the
                // pager_playback() function for additional information.
                uint nJRec;         // Number of Journal Records
                if (nJRec == 0 && pPager.journalHdr + JOURNAL_HDR_SZ(pPager) >= pPager.journalOff)
                    nJRec = (uint)((szJ - pPager.journalOff) / JOURNAL_PG_SZ(pPager));
                for (uint ii = 0; rc == SQLITE.OK && ii < nJRec && pPager.journalOff < szJ; ii++)
                    rc = pager_playback_one_page(pPager, ref pPager.journalOff, pDone, 1, 1);
                Debug.Assert(rc != SQLITE.DONE);
            }
            Debug.Assert(rc != SQLITE.OK || pPager.journalOff >= szJ);
            // Finally,  rollback pages from the sub-journal.  Page that were previously rolled back out of the main journal (and are hence in pDone)
            // will be skipped.  Out-of-range pages are also skipped.
            if (pSavepoint != null)
            {
                long offset = pSavepoint.iSubRec * (4 + pPager.pageSize);
                if (pagerUseWal(pPager))
                    rc = sqlite3WalSavepointUndo(pPager.pWal, pSavepoint.aWalData);
                for (var ii = pSavepoint.iSubRec; rc == SQLITE.OK && ii < pPager.nSubRec; ii++)
                {
                    Debug.Assert(offset == ii * (4 + pPager.pageSize));
                    rc = pager_playback_one_page(pPager, ref offset, pDone, 0, 1);
                }
                Debug.Assert(rc != SQLITE.DONE);
            }
            sqlite3BitvecDestroy(ref pDone);
            if (rc == SQLITE.OK)
                pPager.journalOff = (int)szJ;
            return rc;
        }

        internal static void sqlite3PagerSetCachesize(Pager pPager, int mxPage)
        {
            sqlite3PcacheSetCachesize(pPager.pPCache, mxPage);
        }

#if !SQLITE_OMIT_PAGER_PRAGMAS
        internal static void sqlite3PagerSetSafetyLevel(Pager pPager, int level, int bFullFsync, int bCkptFullFsync)
        {
            Debug.Assert(level >= 1 && level <= 3);
            pPager.noSync = (level == 1 || pPager.tempFile);
            pPager.fullSync = (level == 3 && !pPager.tempFile);
            if (pPager.noSync)
            {
                pPager.syncFlags = 0;
                pPager.ckptSyncFlags = 0;
            }
            else if (bFullFsync != 0)
            {
                pPager.syncFlags = SQLITE_SYNC_FULL;
                pPager.ckptSyncFlags = SQLITE_SYNC_FULL;
            }
            else if (bCkptFullFsync != 0)
            {
                pPager.syncFlags = SQLITE_SYNC_NORMAL;
                pPager.ckptSyncFlags = SQLITE_SYNC_FULL;
            }
            else
            {
                pPager.syncFlags = SQLITE_SYNC_NORMAL;
                pPager.ckptSyncFlags = SQLITE_SYNC_NORMAL;
            }
        }
#endif

        internal static int pagerOpentemp(Pager pPager, ref sqlite3_file pFile, int vfsFlags)
        {
            vfsFlags |= SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_EXCLUSIVE | SQLITE_OPEN_DELETEONCLOSE;
            int dummy = 0;
            var rc = sqlite3OsOpen(pPager.pVfs, null, pFile, vfsFlags, ref dummy);
            Debug.Assert(rc != SQLITE.OK || pFile.isOpen);
            return rc;
        }


        internal static void sqlite3PagerSetBusyhandler(Pager pPager, Func<object, int> xBusyHandler, object pBusyHandlerArg)
        {
            pPager.xBusyHandler = xBusyHandler;
            pPager.pBusyHandlerArg = pBusyHandlerArg;
        }

        internal static SQLITE sqlite3PagerSetPagesize(Pager pPager, ref uint pPageSize, int nReserve)
        {
            // It is not possible to do a full assert_pager_state() here, as this function may be called from within PagerOpen(), before the state
            // of the Pager object is internally consistent.
            // At one point this function returned an error if the pager was in PAGER_ERROR state. But since PAGER_ERROR state guarantees that
            // there is at least one outstanding page reference, this function is a no-op for that case anyhow.
            var rc = SQLITE.OK;
            var pageSize = pPageSize;
            Debug.Assert(pageSize == 0 || (pageSize >= 512 && pageSize <= SQLITE_MAX_PAGE_SIZE));
            if ((pPager.memDb == 0 || pPager.dbSize == 0) && sqlite3PcacheRefCount(pPager.pPCache) == 0 && pageSize != 0 && pageSize != (uint)pPager.pageSize)
            {
                long nByte = 0;
                if (pPager.eState > PAGER.OPEN && pPager.fd.isOpen)
                    rc = sqlite3OsFileSize(pPager.fd, ref nByte);
                if (rc == SQLITE.OK)
                {
                    pager_reset(pPager);
                    pPager.dbSize = (Pgno)(nByte / pageSize);
                    pPager.pageSize = (int)pageSize;
                    sqlite3PageFree(ref pPager.pTmpSpace);
                    pPager.pTmpSpace = new byte[pageSize];
                    sqlite3PcacheSetPageSize(pPager.pPCache, (int)pageSize);
                }
            }
            pPageSize = (uint)pPager.pageSize;
            if (rc == SQLITE.OK)
            {
                if (nReserve < 0)
                    nReserve = pPager.nReserve;
                Debug.Assert(nReserve >= 0 && nReserve < 1000);
                pPager.nReserve = (short)nReserve;
                pagerReportSize(pPager);
            }
            return rc;
        }

        internal static byte[] sqlite3PagerTempSpace(Pager pPager)
        {
            return pPager.pTmpSpace;
        }

        internal static Pgno sqlite3PagerMaxPageCount(Pager pPager, int mxPage)
        {
            if (mxPage > 0)
                pPager.mxPgno = (Pgno)mxPage;
            Debug.Assert(pPager.eState != PAGER.OPEN);     // Called only by OP_MaxPgcnt
            Debug.Assert(pPager.mxPgno >= pPager.dbSize);  // OP_MaxPgcnt enforces this
            return pPager.mxPgno;
        }

        internal static SQLITE sqlite3PagerReadFileheader(Pager pPager, int N, byte[] pDest)
        {
            var rc = SQLITE.OK;
            Array.Clear(pDest, 0, N);
            Debug.Assert(pPager.fd.isOpen || pPager.tempFile);
            // This routine is only called by btree immediately after creating the Pager object.  There has not been an opportunity to transition
            // to WAL mode yet.
            Debug.Assert(!pagerUseWal(pPager));
            if (pPager.fd.isOpen)
            {
                IOTRACE("DBHDR %p 0 %d\n", pPager, N);
                rc = sqlite3OsRead(pPager.fd, pDest, N, 0);
                if (rc == SQLITE.IOERR_SHORT_READ)
                    rc = SQLITE.OK;
            }
            return rc;
        }

        internal static void sqlite3PagerPagecount(Pager pPager, out Pgno pnPage)
        {
            Debug.Assert(pPager.eState >= PAGER.READER);
            Debug.Assert(pPager.eState != PAGER.WRITER_FINISHED);
            pnPage = pPager.dbSize;
        }

        internal static SQLITE pager_wait_on_lock(Pager pPager, LOCK locktype)
        {
            // Check that this is either a no-op (because the requested lock is already held, or one of the transistions that the busy-handler
            // may be invoked during, according to the comment above sqlite3PagerSetBusyhandler().
            Debug.Assert((pPager.eLock >= locktype) || (pPager.eLock == LOCK.NO && locktype == LOCK.SHARED) || (pPager.eLock == LOCK.RESERVED && locktype == LOCK.EXCLUSIVE));
            SQLITE rc;
            do
                rc = pagerLockDb(pPager, locktype);
            while (rc == SQLITE.BUSY && pPager.xBusyHandler(pPager.pBusyHandlerArg) != 0);
            return rc;
        }

#if DEBUG
        internal static void assertTruncateConstraintCb(PgHdr pPg) { Debug.Assert((pPg.flags & PgHdr.PGHDR.DIRTY) != 0); Debug.Assert(!subjRequiresPage(pPg) || pPg.pgno <= pPg.pPager.dbSize); }
        internal static void assertTruncateConstraint(Pager pPager) { sqlite3PcacheIterateDirty(pPager.pPCache, assertTruncateConstraintCb); }
#else
        internal static void assertTruncateConstraintCb(PgHdr pPg) { }
        internal static void assertTruncateConstraint(Pager pPager) { }
#endif

        internal static void sqlite3PagerTruncateImage(Pager pPager, uint nPage)
        {
            Debug.Assert(pPager.dbSize >= nPage);
            Debug.Assert(pPager.eState >= PAGER.WRITER_CACHEMOD);
            pPager.dbSize = nPage;
            assertTruncateConstraint(pPager);
        }

        internal static int pagerSyncHotJournal(Pager pPager)
        {
            var rc = SQLITE.OK;
            if (!pPager.noSync)
                rc = sqlite3OsSync(pPager.jfd, SQLITE_SYNC_NORMAL);
            if (rc == SQLITE.OK)
                rc = sqlite3OsFileSize(pPager.jfd, ref pPager.journalHdr);
            return rc;
        }

        internal static int sqlite3PagerClose(Pager pPager)
        {
            var pTmp = pPager.pTmpSpace;
            sqlite3BeginBenignMalloc();
            pPager.exclusiveMode = false;
            //#if !SQLITE_OMIT_WAL
            //            sqlite3WalClose(pPager->pWal, pPager->ckptSyncFlags, pPager->pageSize, pTmp);
            //            pPager.pWal = 0;
            //#endif
            pager_reset(pPager);
            if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
1 == pPager.memDb
#endif
)
                pager_unlock(pPager);
            else
            {
                // If it is open, sync the journal file before calling UnlockAndRollback.
                // If this is not done, then an unsynced portion of the open journal file may be played back into the database. If a power failure occurs 
                // while this is happening, the database could become corrupt.
                // If an error occurs while trying to sync the journal, shift the pager into the ERROR state. This causes UnlockAndRollback to unlock the
                // database and close the journal file without attempting to roll it back or finalize it. The next database user will have to do hot-journal
                // rollback before accessing the database file.
                if (isOpen(pPager.jfd))
                    pager_error(pPager, pagerSyncHotJournal(pPager));
                pagerUnlockAndRollback(pPager);
            }
            sqlite3EndBenignMalloc();
            PAGERTRACE("CLOSE %d\n", PAGERID(pPager));
            IOTRACE("CLOSE %p\n", pPager);
            sqlite3OsClose(pPager.jfd);
            sqlite3OsClose(pPager.fd);
            sqlite3PcacheClose(pPager.pPCache);
#if SQLITE_HAS_CODEC
            if (pPager.xCodecFree != null)
                pPager.xCodecFree(ref pPager.pCodec);
#endif
            Debug.Assert(null == pPager.aSavepoint && !pPager.pInJournal);
            Debug.Assert(!pPager.jfd.isOpen && !pPager.sjfd.isOpen);
            return SQLITE.OK;
        }

        internal static Pgno sqlite3PagerPagenumber(DbPage pPg) { return pPg.pgno; }

        internal static void sqlite3PagerRef(DbPage pPg) { sqlite3PcacheRef(pPg); }

        internal static int syncJournal(Pager pPager, int newHdr)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(pPager));
            Debug.Assert(!pagerUseWal(pPager));
            rc = sqlite3PagerExclusiveLock(pPager);
            if (rc != SQLITE.OK)
                return rc;
            if (!pPager.noSync)
            {
                Debug.Assert(!pPager.tempFile);
                if (pPager.jfd.isOpen && pPager.journalMode != PAGER_JOURNALMODE_MEMORY)
                {
                    var iDc = sqlite3OsDeviceCharacteristics(pPager.fd);
                    Debug.Assert(pPager.jfd.isOpen);
                    if (0 == (iDc & SQLITE_IOCAP_SAFE_APPEND))
                    {
                        // This block deals with an obscure problem. If the last connection that wrote to this database was operating in persistent-journal
                        // mode, then the journal file may at this point actually be larger than Pager.journalOff bytes. If the next thing in the journal
                        // file happens to be a journal-header (written as part of the previous connection's transaction), and a crash or power-failure
                        // occurs after nRec is updated but before this connection writes anything else to the journal file (or commits/rolls back its
                        // transaction), then SQLite may become confused when doing the hot-journal rollback following recovery. It may roll back all
                        // of this connections data, then proceed to rolling back the old, out-of-date data that follows it. Database corruption.
                        // To work around this, if the journal file does appear to contain a valid header following Pager.journalOff, then write a 0x00
                        // byte to the start of it to prevent it from being recognized.
                        // Variable iNextHdrOffset is set to the offset at which this problematic header will occur, if it exists. aMagic is used
                        // as a temporary buffer to inspect the first couple of bytes of the potential journal header.
                        var zHeader = new byte[aJournalMagic.Length + 4];
                        aJournalMagic.CopyTo(zHeader, 0);
                        put32bits(zHeader, aJournalMagic.Length, pPager.nRec);
                        var iNextHdrOffset = journalHdrOffset(pPager);
                        var aMagic = new byte[8];
                        rc = sqlite3OsRead(pPager.jfd, aMagic, 8, iNextHdrOffset);
                        if (rc == SQLITE.OK && 0 == ArrayEx.Compare(aMagic, aJournalMagic, 8))
                        {
                            var zerobyte = new byte[1];
                            rc = sqlite3OsWrite(pPager.jfd, zerobyte, 1, iNextHdrOffset);
                        }
                        if (rc != SQLITE.OK && rc != SQLITE.IOERR_SHORT_READ)
                            return rc;
                        // Write the nRec value into the journal file header. If in full-synchronous mode, sync the journal first. This ensures that
                        // all data has really hit the disk before nRec is updated to mark it as a candidate for rollback.
                        // This is not required if the persistent media supports the SAFE_APPEND property. Because in this case it is not possible
                        // for garbage data to be appended to the file, the nRec field is populated with 0xFFFFFFFF when the journal header is written
                        // and never needs to be updated.
                        if (pPager.fullSync && 0 == (iDc & SQLITE_IOCAP_SEQUENTIAL))
                        {
                            PAGERTRACE("SYNC journal of %d\n", PAGERID(pPager));
                            IOTRACE("JSYNC %p\n", pPager);
                            rc = sqlite3OsSync(pPager.jfd, pPager.syncFlags);
                            if (rc != SQLITE.OK)
                                return rc;
                        }
                        IOTRACE("JHDR %p %lld\n", pPager, pPager.journalHdr);
                        rc = sqlite3OsWrite(pPager.jfd, zHeader, zHeader.Length, pPager.journalHdr);
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                    if (0 == (iDc & SQLITE_IOCAP_SEQUENTIAL))
                    {
                        PAGERTRACE("SYNC journal of %d\n", PAGERID(pPager));
                        IOTRACE("JSYNC %p\n", pPager);
                        rc = sqlite3OsSync(pPager.jfd, pPager.syncFlags | (pPager.syncFlags == SQLITE_SYNC_FULL ? SQLITE_SYNC_DATAONLY : 0));
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                    pPager.journalHdr = pPager.journalOff;
                    if (newHdr != 0 && 0 == (iDc & SQLITE_IOCAP_SAFE_APPEND))
                    {
                        pPager.nRec = 0;
                        rc = writeJournalHdr(pPager);
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                }
                else
                    pPager.journalHdr = pPager.journalOff;
            }
            // Unless the pager is in noSync mode, the journal file was just successfully synced. Either way, clear the PGHDR_NEED_SYNC flag on 
            // all pages.
            sqlite3PcacheClearSyncFlags(pPager.pPCache);
            pPager.eState = PAGER.WRITER_DBMOD;
            Debug.Assert(assert_pager_state(pPager));
            return SQLITE.OK;
        }

        internal static SQLITE pager_write_pagelist(Pager pPager, PgHdr pList)
        {
            var rc = SQLITE.OK;
            // This function is only called for rollback pagers in WRITER_DBMOD state. 
            Debug.Assert(!pagerUseWal(pPager));
            Debug.Assert(pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(pPager.eLock == LOCK.EXCLUSIVE);
            // If the file is a temp-file has not yet been opened, open it now. It is not possible for rc to be other than SQLITE_OK if this branch
            // is taken, as pager_wait_on_lock() is a no-op for temp-files.
            if (!pPager.fd.isOpen)
            {
                Debug.Assert(pPager.tempFile && rc == SQLITE.OK);
                rc = pagerOpentemp(pPager, ref pPager.fd, (int)pPager.vfsFlags);
            }
            // Before the first write, give the VFS a hint of what the final file size will be.
            Debug.Assert(rc != SQLITE.OK || pPager.fd.isOpen);
            if (rc == SQLITE.OK && pPager.dbSize > pPager.dbHintSize)
            {
                long szFile = pPager.pageSize * (long)pPager.dbSize;
                sqlite3OsFileControl(pPager.fd, SQLITE_FCNTL_SIZE_HINT, ref szFile);
                pPager.dbHintSize = pPager.dbSize;
            }
            while (rc == SQLITE.OK && pList)
            {
                var pgno = pList.pgno;
                // If there are dirty pages in the page cache with page numbers greater than Pager.dbSize, this means sqlite3PagerTruncateImage() was called to
                // make the file smaller (presumably by auto-vacuum code). Do not write any such pages to the file.
                // Also, do not write out any page that has the PGHDR_DONT_WRITE flag set (set by sqlite3PagerDontWrite()).
                if (pList.pgno <= pPager.dbSize && 0 == (pList.flags & PGHDR_DONT_WRITE))
                {
                    Debug.Assert((pList.flags & PGHDR_NEED_SYNC) == 0);
                    if (pList.pgno == 1)
                        pager_write_changecounter(pList);
                    // Encode the database
                    byte[] pData = null;                                    // Data to write
                    if (CODEC2(pPager, pList.pData, pgno, SQLITE_ENCRYPT_WRITE_CTX, ref pData))
                        return SQLITE_NOMEM;
                    // Write out the page data.
                    long offset = (pList.pgno - 1) * (long)pPager.pageSize;   // Offset to write
                    rc = sqlite3OsWrite(pPager.fd, pData, pPager.pageSize, offset);
                    // If page 1 was just written, update Pager.dbFileVers to match the value now stored in the database file. If writing this
                    // page caused the database file to grow, update dbFileSize.
                    if (pgno == 1)
                        Buffer.BlockCopy(pData, 24, pPager.dbFileVers, 0, pPager.dbFileVers.Length);
                    if (pgno > pPager.dbFileSize)
                        pPager.dbFileSize = pgno;
                    // Update any backup objects copying the contents of this pager.
                    sqlite3BackupUpdate(pPager.pBackup, pgno, pList.pData);
                    PAGERTRACE("STORE %d page %d hash(%08x)\n",
                    PAGERID(pPager), pgno, pager_pagehash(pList));
                    IOTRACE("PGOUT %p %d\n", pPager, pgno);
                }
                else
                    PAGERTRACE("NOSTORE %d page %d\n", PAGERID(pPager), pgno);
                pager_set_pagehash(pList);
                pList = pList.pDirty;
            }
            return rc;
        }

        internal static SQLITE openSubJournal(Pager pPager)
        {
            var rc = SQLITE.OK;
            if (!pPager.sjfd.isOpen)
            {
                if (pPager.journalMode == JOURNALMODE.MEMORY || pPager.subjInMemory != 0)
                    sqlite3MemJournalOpen(pPager.sjfd);
                else
                    rc = pagerOpentemp(pPager, ref pPager.sjfd, SQLITE_OPEN_SUBJOURNAL);
            }
            return rc;
        }

        internal static SQLITE subjournalPage(PgHdr pPg)
        {
            var rc = SQLITE.OK;
            var pPager = pPg.pPager;
            if (pPager.journalMode != JOURNALMODE.OFF)
            {
                // Open the sub-journal, if it has not already been opened
                Debug.Assert(pPager.useJournal != 0);
                Debug.Assert(pPager.jfd.isOpen || pagerUseWal(pPager));
                Debug.Assert(pPager.sjfd.isOpen || pPager.nSubRec == 0);
                Debug.Assert(pagerUseWal(pPager) || pageInJournal(pPg) || pPg.pgno > pPager.dbOrigSize);
                rc = openSubJournal(pPager);
                // If the sub-journal was opened successfully (or was already open), write the journal record into the file. 
                if (rc == SQLITE.OK)
                {
                    var pData = pPg.pData;
                    long offset = pPager.nSubRec * (4 + pPager.pageSize);
                    byte[] pData2 = null;
                    if (CODEC2(pPager, pData, pPg.pgno, SQLITE_ENCRYPT_READ_CTX, ref pData2))
                        return SQLITE.NOMEM;
                    PAGERTRACE("STMT-JOURNAL %d page %d\n", PAGERID(pPager), pPg.pgno);
                    rc = write32bits(pPager.sjfd, offset, pPg.pgno);
                    if (rc == SQLITE.OK)
                        rc = sqlite3OsWrite(pPager.sjfd, pData2, pPager.pageSize, offset + 4);
                }
            }
            if (rc == SQLITE.OK)
            {
                pPager.nSubRec++;
                Debug.Assert(pPager.nSavepoint > 0);
                rc = addToSavepointBitvecs(pPager, pPg.pgno);
            }
            return rc;
        }

        internal static SQLITE pagerStress(object p, PgHdr pPg)
        {
            var pPager = (Pager)p;
            var rc = SQLITE.OK;
            Debug.Assert(pPg.pPager == pPager);
            Debug.Assert((pPg.flags & PGHDR_DIRTY) != 0);
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
            if (pagerUseWal(pPager))
            {
                // Write a single frame for this page to the log.
                if (subjRequiresPage(pPg))
                    rc = subjournalPage(pPg);
                if (rc == SQLITE.OK)
                    rc = pagerWalFrames(pPager, pPg, 0, 0, 0);
            }
            else
            {
                // Sync the journal file if required. 
                if ((pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 || pPager.eState == PAGER.WRITER_CACHEMOD)
                    rc = syncJournal(pPager, 1);
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
                    Debug.Assert((pPg.flags & PGHDR_NEED_SYNC) == 0);
                    rc = pager_write_pagelist(pPager, pPg);
                }
            }
            // Mark the page as clean.
            if (rc == SQLITE.OK)
            {
                PAGERTRACE("STRESS %d page %d\n", PAGERID(pPager), pPg.pgno);
                sqlite3PcacheMakeClean(pPg);
            }
            return pager_error(pPager, rc);
        }

        internal static SQLITE sqlite3PagerOpen(sqlite3_vfs pVfs, out Pager ppPager, string zFilename, int nExtra, int flags, int vfsFlags, Action<PgHdr> xReinit)
        {
            ushort pPtr;
            Pager pPager = null;     // Pager object to allocate and return
            byte tempFile = 0;         // True for temp files (incl. in-memory files) 
            byte memDb = 0;            // True if this is an in-memory file
            bool readOnly = false;   // True if this is a read-only file
            int journalFileSize;
            StringBuilder zPathname = null; // Full path to database file
            int nPathname = 0;       // Number of bytes in zPathname
            bool useJournal = (flags & PAGER_OMIT_JOURNAL) == 0; // False to omit journal
            bool noReadlock = (flags & PAGER_NO_READLOCK) != 0;  // True to omit read-lock
            int pcacheSize = sqlite3PcacheSize();       // Bytes to allocate for PCache
            uint szPageDflt = SQLITE_DEFAULT_PAGE_SIZE;  // Default page size
            string zUri = null;     // URI args to copy
            int nUri = 0;           // Number of bytes of URI args at *zUri
            // Figure out how much space is required for each journal file-handle (there are two of them, the main journal and the sub-journal). This
            // is the maximum space required for an in-memory journal file handle and a regular journal file-handle. Note that a "regular journal-handle"
            // may be a wrapper capable of caching the first portion of the journal file in memory to implement the atomic-write optimization (see
            // source file journal.c).
            int journalFileSize = ROUND8(sqlite3JournalSize(pVfs) > sqlite3MemJournalSize() ? sqlite3JournalSize(pVfs) : sqlite3MemJournalSize()); // Bytes to allocate for each journal fd
            // Set the output variable to NULL in case an error occurs.
            ppPager = null;
#if !SQLITE_OMIT_MEMORYDB
            if ((flags & PAGER_MEMORY) != 0)
            {
                memDb = 1;
                zFilename = null;
            }
#endif
            // Compute and store the full pathname in an allocated buffer pointed to by zPathname, length nPathname. Or, if this is a temporary file,
            // leave both nPathname and zPathname set to 0.
            var rc = SQLITE.OK;
            if (!string.IsNullOrEmpty(zFilename))
            {
                nPathname = pVfs.mxPathname + 1;
                zPathname = new StringBuilder(nPathname * 2);
                rc = sqlite3OsFullPathname(pVfs, zFilename, nPathname, zPathname);
                nPathname = sqlite3Strlen30(zPathname);
                var z = zUri = zFilename;
                nUri = zUri.Length;
                if (rc == SQLITE.OK && nPathname + 8 > pVfs.mxPathname)
                    // This branch is taken when the journal path required by the database being opened will be more than pVfs.mxPathname
                    // bytes in length. This means the database cannot be opened, as it will not be possible to open the journal file or even
                    // check for a hot-journal before reading.
                    rc = SQLITE.CANTOPEN_BKPT();
                if (rc != SQLITE.OK)
                    return rc;
            }
            // Allocate memory for the Pager structure, PCache object, the three file descriptors, the database file name and the journal
            // file name. The layout in memory is as follows:
            //     Pager object                    (sizeof(Pager) bytes)
            //     PCache object                   (sqlite3PcacheSize() bytes)
            //     Database file handle            (pVfs.szOsFile bytes)
            //     Sub-journal file handle         (journalFileSize bytes)
            //     Main journal file handle        (journalFileSize bytes)
            //     Database file name              (nPathname+1 bytes)
            //     Journal file name               (nPathname+8+1 bytes)
            pPager = new Pager();
            pPager.pPCache = new PCache();
            pPager.fd = new sqlite3_file();
            pPager.sjfd = new sqlite3_file();
            pPager.jfd = new sqlite3_file();
            // Fill in the Pager.zFilename and Pager.zJournal buffers, if required.
            if (zPathname != null)
            {
                Debug.Assert(nPathname > 0);
                pPager.zFilename = zPathname.ToString();
                zUri = pPager.zFilename;
                pPager.zJournal = pPager.zFilename + "-journal";
                sqlite3FileSuffix3(pPager.zFilename, pPager.zJournal);
#if !SQLITE_OMIT_WAL
                pPager.zWal = &pPager.zJournal[nPathname + 8 + 1];
                memcpy(pPager.zWal, zPathname, nPathname);
                memcpy(&pPager.zWal[nPathname], "-wal", 4);
                sqlite3FileSuffix3(pPager.zFilename, pPager.zWal);
#endif
            }
            else
                pPager.zFilename = "";
            pPager.pVfs = pVfs;
            pPager.vfsFlags = (uint)vfsFlags;
            // Open the pager file.
            if (!string.IsNullOrEmpty(zFilename))
            {
                var fout = 0; // VFS flags returned by xOpen()
                rc = sqlite3OsOpen(pVfs, zFilename, pPager.fd, vfsFlags, ref fout);
                Debug.Assert(0 == memDb);
                readOnly = (fout & SQLITE_OPEN_READONLY) != 0;
                // If the file was successfully opened for read/write access, choose a default page size in case we have to create the
                // database file. The default page size is the maximum of:
                //    + SQLITE_DEFAULT_PAGE_SIZE,
                //    + The value returned by sqlite3OsSectorSize()
                //    + The largest page size that can be written atomically.
                if (rc == SQLITE.OK && !readOnly)
                {
                    setSectorSize(pPager);
                    Debug.Assert(SQLITE_DEFAULT_PAGE_SIZE <= SQLITE_MAX_DEFAULT_PAGE_SIZE);
                    if (szPageDflt < pPager.sectorSize)
                        szPageDflt = (pPager.sectorSize > SQLITE_MAX_DEFAULT_PAGE_SIZE ? SQLITE_MAX_DEFAULT_PAGE_SIZE : (uint)pPager.sectorSize);
#if SQLITE_ENABLE_ATOMIC_WRITE
                    int iDc = sqlite3OsDeviceCharacteristics(pPager.fd);
                    Debug.Assert(SQLITE_IOCAP_ATOMIC512 == (512 >> 8));
                    Debug.Assert(SQLITE_IOCAP_ATOMIC64K == (65536 >> 8));
                    Debug.Assert(SQLITE_MAX_DEFAULT_PAGE_SIZE <= 65536);
                    for (var ii = szPageDflt; ii <= SQLITE_MAX_DEFAULT_PAGE_SIZE; ii = ii * 2)
                        if (iDc & (SQLITE_IOCAP_ATOMIC | (ii >> 8)))
                            szPageDflt = ii;
#endif
                }
            }
            else
            {
                // If a temporary file is requested, it is not opened immediately. In this case we accept the default page size and delay actually
                // opening the file until the first call to OsWrite().
                // This branch is also run for an in-memory database. An in-memory database is the same as a temp-file that is never written out to
                // disk and uses an in-memory rollback journal.
                tempFile = 1;
                pPager.eState = PAGER_READER;
                pPager.eLock = EXCLUSIVE_LOCK;
                readOnly = (vfsFlags & SQLITE_OPEN_READONLY) != 0;
            }

            // The following call to PagerSetPagesize() serves to set the value of Pager.pageSize and to allocate the Pager.pTmpSpace buffer.
            if (rc == SQLITE.OK)
            {
                Debug.Assert(pPager.memDb == 0);
                rc = sqlite3PagerSetPagesize(pPager, ref szPageDflt, -1);
            }
            // If an error occurred in either of the blocks above, free the Pager structure and close the file.
            if (rc != SQLITE.OK)
            {
                Debug.Assert(null == pPager.pTmpSpace);
                sqlite3OsClose(pPager.fd);
                return rc;
            }
            // Initialize the PCache object.
            Debug.Assert(nExtra < 1000);
            nExtra = ROUND8(nExtra);
            sqlite3PcacheOpen((int)szPageDflt, nExtra, 0 == memDb,
            0 == memDb ? (dxStress)pagerStress : null, pPager, pPager.pPCache);
            PAGERTRACE("OPEN %d %s\n", FILEHANDLEID(pPager.fd), pPager.zFilename);
            IOTRACE("OPEN %p %s\n", pPager, pPager.zFilename);
            pPager.useJournal = (byte)(useJournal ? 1 : 0);
            pPager.noReadlock = (byte)(noReadlock && readOnly ? 1 : 0);
            pPager.mxPgno = SQLITE_MAX_PAGE_COUNT;
#if false
            Debug.Assert(pPager.state == (tempFile != 0 ? PAGER.EXCLUSIVE : PAGER.UNLOCK));
#endif
            pPager.tempFile = tempFile != 0;
            Debug.Assert(tempFile == LOCKINGMODE.NORMAL || tempFile == LOCKINGMODE.EXCLUSIVE);
            Debug.Assert(LOCKINGMODE.EXCLUSIVE == 1);
            pPager.exclusiveMode = tempFile != 0;
            pPager.changeCountDone = pPager.tempFile;
            pPager.memDb = memDb;
            pPager.readOnly = readOnly;
            Debug.Assert(useJournal || pPager.tempFile);
            pPager.noSync = pPager.tempFile;
            pPager.fullSync = pPager.noSync;
            pPager.syncFlags = (byte)(pPager.noSync ? 0 : SQLITE_SYNC_NORMAL);
            pPager.ckptSyncFlags = pPager.syncFlags;
            pPager.nExtra = (ushort)nExtra;
            pPager.journalSizeLimit = SQLITE_DEFAULT_JOURNAL_SIZE_LIMIT;
            Debug.Assert(pPager.fd.isOpen || tempFile != 0);
            setSectorSize(pPager);
            if (!useJournal)
                pPager.journalMode = JOURNALMODE.OFF;
            else if (memDb != 0)
                pPager.journalMode = JOURNALMODE.MEMORY;
            pPager.xReiniter = xReinit;
            ppPager = pPager;
            return SQLITE.OK;
        }

        internal static SQLITE hasHotJournal(Pager pPager, ref int pExists)
        {
            sqlite3_vfs pVfs = pPager.pVfs;
            int exists = 1;               // True if a journal file is present
            int jrnlOpen = isOpen(pPager.jfd) ? 1 : 0;
            Debug.Assert(pPager.useJournal != 0);
            Debug.Assert(isOpen(pPager.fd));
            Debug.Assert(pPager.eState == PAGER_OPEN);
            Debug.Assert(jrnlOpen == 0 || (sqlite3OsDeviceCharacteristics(pPager.jfd) & SQLITE_IOCAP_UNDELETABLE_WHEN_OPEN) != 0);
            pExists = 0;
            var rc = SQLITE.OK;
            if (0 == jrnlOpen)
                rc = sqlite3OsAccess(pVfs, pPager.zJournal, SQLITE_ACCESS_EXISTS, ref exists);
            if (rc == SQLITE_OK && exists != 0)
            {
                int locked = 0;                 // True if some process holds a RESERVED lock
                // Race condition here:  Another process might have been holding the the RESERVED lock and have a journal open at the sqlite3OsAccess()
                // call above, but then delete the journal and drop the lock before we get to the following sqlite3OsCheckReservedLock() call.  If that
                // is the case, this routine might think there is a hot journal when in fact there is none.  This results in a false-positive which will
                // be dealt with by the playback routine.  Ticket #3883.
                rc = sqlite3OsCheckReservedLock(pPager.fd, ref locked);
                if (rc == SQLITE_OK && locked == 0)
                {
                    Pgno nPage = 0;                 // Number of pages in database file
                    // Check the size of the database file. If it consists of 0 pages, then delete the journal file. See the header comment above for
                    // the reasoning here.  Delete the obsolete journal file under a RESERVED lock to avoid race conditions and to avoid violating [H33020].
                    rc = pagerPagecount(pPager, ref nPage);
                    if (rc == SQLITE.OK)
                    {
                        if (nPage == 0)
                        {
                            sqlite3BeginBenignMalloc();
                            if (pagerLockDb(pPager, RESERVED_LOCK) == SQLITE_OK)
                            {
                                sqlite3OsDelete(pVfs, pPager.zJournal, 0);
                                if (!pPager.exclusiveMode)
                                    pagerUnlockDb(pPager, SHARED_LOCK);
                            }
                            sqlite3EndBenignMalloc();
                        }
                        else
                        {
                            // The journal file exists and no other connection has a reserved or greater lock on the database file. Now check that there is
                            // at least one non-zero bytes at the start of the journal file. If there is, then we consider this journal to be hot. If not,
                            // it can be ignored.
                            if (0 == jrnlOpen)
                            {
                                int f = SQLITE_OPEN_READONLY | SQLITE_OPEN_MAIN_JOURNAL;
                                rc = sqlite3OsOpen(pVfs, pPager.zJournal, pPager.jfd, f, ref f);
                            }
                            if (rc == SQLITE.OK)
                            {
                                var first = new byte[1];
                                rc = sqlite3OsRead(pPager.jfd, first, 1, 0);
                                if (rc == SQLITE.IOERR_SHORT_READ)
                                    rc = SQLITE.OK;
                                if (0 == jrnlOpen)
                                    sqlite3OsClose(pPager.jfd);
                                pExists = (first[0] != 0) ? 1 : 0;
                            }
                            else if (rc == SQLITE.CANTOPEN)
                            {
                                // If we cannot open the rollback journal file in order to see if its has a zero header, that might be due to an I/O error, or
                                // it might be due to the race condition.  Either way, assume that the journal is hot. This might be a false positive.  But if it is, then the
                                // automatic journal playback and recovery mechanism will deal with it under an EXCLUSIVE lock where we do not need to
                                // worry so much with race conditions.
                                pExists = 1;
                                rc = SQLITE.OK;
                            }
                        }
                    }
                }
            }
            return rc;
        }

        internal static int sqlite3PagerSharedLock(Pager pPager)
        {
            var rc = SQLITE.OK;
            // This routine is only called from b-tree and only when there are no outstanding pages. This implies that the pager state should either
            // be OPEN or READER. READER is only possible if the pager is or was in exclusive access mode.
            Debug.Assert(sqlite3PcacheRefCount(pPager.pPCache) == 0);
            Debug.Assert(assert_pager_state(pPager));
            Debug.Assert(pPager.eState == PAGER.OPEN || pPager.eState == PAGER.READER);
            if (Check.NEVER(
#if SQLITE_OMIT_MEMORYDB
0!=MEMDB
#else
0 != pPager.memDb
#endif
 && pPager.errCode != 0))
                return pPager.errCode;
            if (!pagerUseWal(pPager) && pPager.eState == PAGER.OPEN)
            {
                int bHotJournal = 1; // True if there exists a hot journal-file
                Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
);
                Debug.Assert(pPager.noReadlock == 0 || pPager.readOnly);
                if (pPager.noReadlock == 0)
                {
                    rc = pager_wait_on_lock(pPager, LOCK.SHARED);
                    if (rc != SQLITE.OK)
                    {
                        Debug.Assert(pPager.eLock == LOCK.NO || pPager.eLock == LOCK.UNKNOWN);
                        goto failed;
                    }
                }
                // If a journal file exists, and there is no RESERVED lock on the database file, then it either needs to be played back or deleted.
                if (pPager.eLock <= LOCK.SHARED)
                    rc = hasHotJournal(pPager, ref bHotJournal);
                if (rc != SQLITE.OK)
                    goto failed;
                if (bHotJournal != 0)
                {
                    // Get an EXCLUSIVE lock on the database file. At this point it is important that a RESERVED lock is not obtained on the way to the
                    // EXCLUSIVE lock. If it were, another process might open the database file, detect the RESERVED lock, and conclude that the
                    // database is safe to read while this process is still rolling the 
                    // hot-journal back.
                    // Because the intermediate RESERVED lock is not requested, any other process attempting to access the database file will get to 
                    // this point in the code and fail to obtain its own EXCLUSIVE lock on the database file.
                    // Unless the pager is in locking_mode=exclusive mode, the lock is downgraded to SHARED_LOCK before this function returns.
                    rc = pagerLockDb(pPager, LOCK.EXCLUSIVE);
                    if (rc != SQLITE.OK)
                        goto failed;
                    // If it is not already open and the file exists on disk, open the journal for read/write access. Write access is required because 
                    // in exclusive-access mode the file descriptor will be kept open and possibly used for a transaction later on. Also, write-access 
                    // is usually required to finalize the journal in journal_mode=persist mode (and also for journal_mode=truncate on some systems).
                    // If the journal does not exist, it usually means that some other connection managed to get in and roll it back before 
                    // this connection obtained the exclusive lock above. Or, it may mean that the pager was in the error-state when this
                    // function was called and the journal file does not exist.
                    if (!pPager.jfd.isOpen)
                    {
                        var pVfs = pPager.pVfs;
                        var bExists = 0;              // True if journal file exists
                        rc = sqlite3OsAccess(pVfs, pPager.zJournal, SQLITE_ACCESS_EXISTS, ref bExists);
                        if (rc == SQLITE.OK && bExists != 0)
                        {
                            var fout = 0;
                            var f = SQLITE_OPEN_READWRITE | SQLITE_OPEN_MAIN_JOURNAL;
                            Debug.Assert(!pPager.tempFile);
                            rc = sqlite3OsOpen(pVfs, pPager.zJournal, pPager.jfd, f, ref fout);
                            Debug.Assert(rc != SQLITE.OK || pPager.jfd.isOpen);
                            if (rc == SQLITE.OK && (fout & SQLITE_OPEN_READONLY) != 0)
                            {
                                rc = SQLITE_CANTOPEN_BKPT();
                                sqlite3OsClose(pPager.jfd);
                            }
                        }
                    }
                    // Playback and delete the journal.  Drop the database write lock and reacquire the read lock. Purge the cache before
                    // playing back the hot-journal so that we don't end up with an inconsistent cache.  Sync the hot journal before playing
                    // it back since the process that crashed and left the hot journal probably did not sync it and we are required to always sync
                    // the journal before playing it back.
                    if (isOpen(pPager.jfd))
                    {
                        Debug.Assert(rc == SQLITE.OK);
                        rc = pagerSyncHotJournal(pPager);
                        if (rc == SQLITE.OK)
                        {
                            rc = pager_playback(pPager, 1);
                            pPager.eState = PAGER_OPEN;
                        }
                    }
                    else if (!pPager.exclusiveMode)
                        pagerUnlockDb(pPager, SHARED_LOCK);
                    if (rc != SQLITE.OK)
                    {
                        // This branch is taken if an error occurs while trying to open or roll back a hot-journal while holding an EXCLUSIVE lock. The
                        // pager_unlock() routine will be called before returning to unlock the file. If the unlock attempt fails, then Pager.eLock must be
                        // set to UNKNOWN_LOCK (see the comment above the #define for UNKNOWN_LOCK above for an explanation). 
                        // In order to get pager_unlock() to do this, set Pager.eState to PAGER_ERROR now. This is not actually counted as a transition
                        // to ERROR state in the state diagram at the top of this file, since we know that the same call to pager_unlock() will very
                        // shortly transition the pager object to the OPEN state. Calling assert_pager_state() would fail now, as it should not be possible
                        // to be in ERROR state when there are zero outstanding page references.
                        pager_error(pPager, rc);
                        goto failed;
                    }
                    Debug.Assert(pPager.eState == PAGER.OPEN);
                    Debug.Assert((pPager.eLock == LOCK.SHARED) || (pPager.exclusiveMode && pPager.eLock > LOCK.SHARED));
                }
                if (!pPager.tempFile && (pPager.pBackup != null || sqlite3PcachePagecount(pPager.pPCache) > 0))
                {
                    // The shared-lock has just been acquired on the database file and there are already pages in the cache (from a previous
                    // read or write transaction).  Check to see if the database has been modified.  If the database has changed, flush the
                    // cache.
                    // Database changes is detected by looking at 15 bytes beginning at offset 24 into the file.  The first 4 of these 16 bytes are
                    // a 32-bit counter that is incremented with each change.  The other bytes change randomly with each file change when
                    // a codec is in use.
                    // There is a vanishingly small chance that a change will not be detected.  The chance of an undetected change is so small that
                    // it can be neglected.
                    Pgno nPage = 0;
                    var dbFileVers = new byte[pPager.dbFileVers.Length];
                    rc = pagerPagecount(pPager, ref nPage);
                    if (rc != 0)
                        goto failed;
                    if (nPage > 0)
                    {
                        IOTRACE("CKVERS %p %d\n", pPager, dbFileVers.Length);
                        rc = sqlite3OsRead(pPager.fd, dbFileVers, dbFileVers.Length, 24);
                        if (rc != SQLITE_OK)
                            goto failed;
                    }
                    else
                        Array.Clear(dbFileVers, 0, dbFileVers.Length);
                    if (ArrayEx.Compare(pPager.dbFileVers, dbFileVers, dbFileVers.Length) != 0)
                        pager_reset(pPager);
                }
                // If there is a WAL file in the file-system, open this database in WAL mode. Otherwise, the following function call is a no-op.
                rc = pagerOpenWalIfPresent(pPager);
#if !SQLITE_OMIT_WAL
                Debug.Assert(pPager.pWal == null || rc == SQLITE.OK);
#endif
            }
            if (pagerUseWal(pPager))
            {
                Debug.Assert(rc == SQLITE.OK);
                rc = pagerBeginReadTransaction(pPager);
            }
            if (pPager.eState == PAGER.OPEN && rc == SQLITE.OK)
                rc = pagerPagecount(pPager, ref pPager.dbSize);

        failed:
            if (rc != SQLITE_OK)
            {
                Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
);
                pager_unlock(pPager);
                Debug.Assert(pPager.eState == PAGER.OPEN);
            }
            else
                pPager.eState = PAGER.READER;
            return rc;
        }


        internal static void pagerUnlockIfUnused(Pager pPager)
        {
            if (sqlite3PcacheRefCount(pPager.pPCache) == 0)
                pagerUnlockAndRollback(pPager);
        }

        internal static SQLITE sqlite3PagerGet(Pager pPager, Pgno pgno, ref DbPage ppPage) { return sqlite3PagerAcquire(pPager, pgno, ref ppPage, 0); }
        internal static SQLITE sqlite3PagerAcquire(Pager pPager, Pgno pgno, ref DbPage ppPage, byte noContent)
        {
            Debug.Assert(pPager.eState >= PAGER.READER);
            Debug.Assert(assert_pager_state(pPager));
            if (pgno == 0)
                return SQLITE_CORRUPT_BKPT();
            // If the pager is in the error state, return an error immediately.  Otherwise, request the page from the PCache layer.
            var rc = (pPager.errCode != SQLITE.OK ? pPager.errCode : sqlite3PcacheFetch(pPager.pPCache, pgno, 1, ref ppPage));
            PgHdr pPg = null;
            if (rc != SQLITE_OK)
            {
                // Either the call to sqlite3PcacheFetch() returned an error or the pager was already in the error-state when this function was called.
                // Set pPg to 0 and jump to the exception handler.  */
                pPg = null;
                goto pager_acquire_err;
            }
            Debug.Assert((ppPage).pgno == pgno);
            Debug.Assert((ppPage).pPager == pPager || (ppPage).pPager == null);
            if ((ppPage).pPager != null && 0 == noContent)
            {
                // In this case the pcache already contains an initialized copy of the page. Return without further ado.
                Debug.Assert(pgno <= PAGER_MAX_PGNO && pgno != PAGER_MJ_PGNO(pPager));
                PAGER_INCR(ref pPager.nHit);
                return SQLITE.OK;
            }
            else
            {
                // The pager cache has created a new page. Its content needs to be initialized.
                pPg = ppPage;
                pPg.pPager = pPager;
                pPg.pExtra = new MemPage();
                // The maximum page number is 2^31. Return SQLITE_CORRUPT if a page number greater than this, or the unused locking-page, is requested.
                if (pgno > PAGER_MAX_PGNO || pgno == PAGER_MJ_PGNO(pPager))
                {
                    rc = SQLITE_CORRUPT_BKPT();
                    goto pager_acquire_err;
                }
                if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
pPager.memDb != 0
#endif
 || pPager.dbSize < pgno || noContent != 0 || !pPager.fd.isOpen)
                {
                    if (pgno > pPager.mxPgno)
                    {
                        rc = SQLITE_FULL;
                        goto pager_acquire_err;
                    }
                    if (noContent != 0)
                    {
                        // Failure to set the bits in the InJournal bit-vectors is benign. It merely means that we might do some extra work to journal a
                        // page that does not need to be journaled.  Nevertheless, be sure to test the case where a malloc error occurs while trying to set
                        // a bit in a bit vector.
                        sqlite3BeginBenignMalloc();
                        if (pgno <= pPager.dbOrigSize)
                            sqlite3BitvecSet(pPager.pInJournal, pgno);
                        addToSavepointBitvecs(pPager, pgno);
                        sqlite3EndBenignMalloc();
                    }
                    Array.Clear(pPg.pData, 0, pPager.pageSize);
                    IOTRACE("ZERO %p %d\n", pPager, pgno);
                }
                else
                {
                    Debug.Assert(pPg.pPager == pPager);
                    rc = readDbPage(pPg);
                    if (rc != SQLITE.OK)
                        goto pager_acquire_err;
                }
                pager_set_pagehash(pPg);
            }
            return SQLITE.OK;
        pager_acquire_err:
            Debug.Assert(rc != SQLITE.OK);
            if (pPg != null)
                sqlite3PcacheDrop(pPg);
            pagerUnlockIfUnused(pPager);
            ppPage = null;
            return rc;
        }

        internal static DbPage sqlite3PagerLookup(Pager pPager, Pgno pgno)
        {
            PgHdr pPg = null;
            Debug.Assert(pPager != null);
            Debug.Assert(pgno != 0);
            Debug.Assert(pPager.pPCache != null);
            Debug.Assert(pPager.eState >= PAGER.READER && pPager.eState != PAGER.ERROR);
            sqlite3PcacheFetch(pPager.pPCache, pgno, 0, ref pPg);
            return pPg;
        }


        internal static void sqlite3PagerUnref(DbPage pPg)
        {
            if (pPg != null)
            {
                var pPager = pPg.pPager;
                sqlite3PcacheRelease(pPg);
                pagerUnlockIfUnused(pPager);
            }
        }

        internal static SQLITE pager_open_journal(Pager pPager)
        {
            var rc = SQLITE.OK;
            var pVfs = pPager.pVfs;
            Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(pPager));
            Debug.Assert(pPager.pInJournal == null);

            // If already in the error state, this function is a no-op.  But on the other hand, this routine is never called if we are already in
            // an error state. */
            if (Check.NEVER(pPager.errCode) != 0)
                return pPager.errCode;
            if (!pagerUseWal(pPager) && pPager.journalMode != JOURNALMODE.OFF)
            {
                pPager.pInJournal = sqlite3BitvecCreate(pPager.dbSize);
                // Open the journal file if it is not already open.
                if (!pPager.jfd.isOpen)
                {
                    if (pPager.journalMode == JOURNALMODE.MEMORY)
                        sqlite3MemJournalOpen(pPager.jfd);
                    else
                    {
                        int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | (pPager.tempFile ? SQLITE_OPEN_DELETEONCLOSE | SQLITE_OPEN_TEMP_JOURNAL : SQLITE_OPEN_MAIN_JOURNAL);
#if SQLITE_ENABLE_ATOMIC_WRITE
                        rc = sqlite3JournalOpen(pVfs, pPager.zJournal, pPager.jfd, flags, jrnlBufferSize(pPager));
#else
                        var int0 = 0;
                        rc = sqlite3OsOpen(pVfs, pPager.zJournal, pPager.jfd, flags, ref int0);
#endif
                    }
                    Debug.Assert(rc != SQLITE.OK || pPager.jfd.isOpen);
                }

                // Write the first journal header to the journal file and open the sub-journal if necessary.
                if (rc == SQLITE.OK)
                {
                    // TODO: Check if all of these are really required.
                    pPager.nRec = 0;
                    pPager.journalOff = 0;
                    pPager.setMaster = 0;
                    pPager.journalHdr = 0;
                    rc = writeJournalHdr(pPager);
                }
            }
            if (rc != SQLITE.OK)
            {
                sqlite3BitvecDestroy(ref pPager.pInJournal);
                pPager.pInJournal = null;
            }
            else
            {
                Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED);
                pPager.eState = PAGER.WRITER_CACHEMOD;
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerBegin(Pager pPager, bool exFlag, int subjInMemory)
        {
            if (pPager.errCode != 0)
                return pPager.errCode;
            Debug.Assert(pPager.eState >= PAGER.READER && pPager.eState < PAGER.ERROR);
            pPager.subjInMemory = (byte)subjInMemory;
            var rc = SQLITE.OK;
            if (ALWAYS(pPager.eState == PAGER.READER))
            {
                Debug.Assert(pPager.pInJournal == null);
                if (pagerUseWal(pPager))
                {
                    // If the pager is configured to use locking_mode=exclusive, and an exclusive lock on the database is not already held, obtain it now.
                    if (pPager.exclusiveMode && sqlite3WalExclusiveMode(pPager.pWal, -1))
                    {
                        rc = pagerLockDb(pPager, LOCK.EXCLUSIVE);
                        if (rc != SQLITE.OK)
                            return rc;
                        sqlite3WalExclusiveMode(pPager.pWal, 1);
                    }
                    // Grab the write lock on the log file. If successful, upgrade to PAGER_RESERVED state. Otherwise, return an error code to the caller.
                    // The busy-handler is not invoked if another connection already holds the write-lock. If possible, the upper layer will call it.
                    rc = sqlite3WalBeginWriteTransaction(pPager.pWal);
                }
                else
                {
                    // Obtain a RESERVED lock on the database file. If the exFlag parameter is true, then immediately upgrade this to an EXCLUSIVE lock. The
                    // busy-handler callback can be used when upgrading to the EXCLUSIVE lock, but not when obtaining the RESERVED lock.
                    rc = pagerLockDb(pPager, LOCK.RESERVED);
                    if (rc == SQLITE.OK && exFlag)
                        rc = pager_wait_on_lock(pPager, LOCK.EXCLUSIVE);
                }
                if (rc == SQLITE.OK)
                {
                    // Change to WRITER_LOCKED state.
                    // WAL mode sets Pager.eState to PAGER_WRITER_LOCKED or CACHEMOD when it has an open transaction, but never to DBMOD or FINISHED.
                    // This is because in those states the code to roll back savepoint transactions may copy data from the sub-journal into the database 
                    // file as well as into the page cache. Which would be incorrect in WAL mode.
                    pPager.eState = PAGER_WRITER_LOCKED;
                    pPager.dbHintSize = pPager.dbSize;
                    pPager.dbFileSize = pPager.dbSize;
                    pPager.dbOrigSize = pPager.dbSize;
                    pPager.journalOff = 0;
                }
                Debug.Assert(rc == SQLITE.OK || pPager.eState == PAGER.READER);
                Debug.Assert(rc != SQLITE.OK || pPager.eState == PAGER.WRITER_LOCKED);
                Debug.Assert(assert_pager_state(pPager));
            }
            PAGERTRACE("TRANSACTION %d\n", PAGERID(pPager));
            return rc;
        }

        internal static SQLITE pager_write(PgHdr pPg)
        {
            var pData = pPg.pData;
            var pPager = pPg.pPager;
            var rc = SQLITE.OK;
            // This routine is not called unless a write-transaction has already been started. The journal file may or may not be open at this point.
            // It is never called in the ERROR state.
            Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED || pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(pPager));
            // If an error has been previously detected, report the same error again. This should not happen, but the check provides robustness. 
            if (Check.NEVER(pPager.errCode) != 0)
                return pPager.errCode;
            // Higher-level routines never call this function if database is not writable.  But check anyway, just for robustness.
            if (Check.NEVER(pPager.readOnly))
                return SQLITE.PERM;
#if SQLITE_CHECK_PAGES
CHECK_PAGE(pPg);
#endif
            // The journal file needs to be opened. Higher level routines have already obtained the necessary locks to begin the write-transaction, but the
            // rollback journal might not yet be open. Open it now if this is the case.
            //
            // This is done before calling sqlite3PcacheMakeDirty() on the page. Otherwise, if it were done after calling sqlite3PcacheMakeDirty(), then
            // an error might occur and the pager would end up in WRITER_LOCKED state with pages marked as dirty in the cache.
            if (pPager.eState == PAGER.WRITER_LOCKED)
            {
                rc = pager_open_journal(pPager);
                if (rc != SQLITE.OK)
                    return rc;
            }
            Debug.Assert(pPager.eState >= PAGER.WRITER_CACHEMOD);
            Debug.Assert(assert_pager_state(pPager));
            // Mark the page as dirty.  If the page has already been written to the journal then we can return right away.
            sqlite3PcacheMakeDirty(pPg);
            if (pageInJournal(pPg) && !subjRequiresPage(pPg))
                Debug.Assert(!pagerUseWal(pPager));
            else
            {
                // The transaction journal now exists and we have a RESERVED or an EXCLUSIVE lock on the main database file.  Write the current page to
                // the transaction journal if it is not there already.
                if (!pageInJournal(pPg) && !pagerUseWal(pPager))
                {
                    Debug.Assert(!pagerUseWal(pPager));
                    if (pPg.pgno <= pPager.dbOrigSize && pPager.jfd.isOpen)
                    {
                        var iOff = pPager.journalOff;
                        // We should never write to the journal file the page that contains the database locks.  The following Debug.Assert verifies that we do not.
                        Debug.Assert(pPg.pgno != ((PENDING_BYTE / (pPager.pageSize)) + 1));
                        Debug.Assert(pPager.journalHdr <= pPager.journalOff);
                        byte[] pData2 = null;
                        if (CODEC2(pPager, pData, pPg.pgno, SQLITE_ENCRYPT_READ_CTX, ref pData2))
                            return SQLITE.NOMEM;
                        var cksum = pager_cksum(pPager, pData2);
                        // Even if an IO or diskfull error occurred while journalling the page in the block above, set the need-sync flag for the page.
                        // Otherwise, when the transaction is rolled back, the logic in playback_one_page() will think that the page needs to be restored
                        // in the database file. And if an IO error occurs while doing so, then corruption may follow.
                        pPg.flags |= PGHDR_NEED_SYNC;
                        rc = write32bits(pPager.jfd, iOff, pPg.pgno);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = sqlite3OsWrite(pPager.jfd, pData2, pPager.pageSize, iOff + 4);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = write32bits(pPager.jfd, iOff + pPager.pageSize + 4, cksum);
                        if (rc != SQLITE.OK)
                            return rc;
                        IOTRACE("JOUT %p %d %lld %d\n", pPager, pPg.pgno,
                        pPager.journalOff, pPager.pageSize);
                        PAGERTRACE("JOURNAL %d page %d needSync=%d hash(%08x)\n",
                        PAGERID(pPager), pPg.pgno, (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0, pager_pagehash(pPg));
                        pPager.journalOff += 8 + pPager.pageSize;
                        pPager.nRec++;
                        Debug.Assert(pPager.pInJournal != null);
                        rc = sqlite3BitvecSet(pPager.pInJournal, pPg.pgno);
                        Debug.Assert(rc == SQLITE.OK || rc == SQLITE.NOMEM);
                        rc |= addToSavepointBitvecs(pPager, pPg.pgno);
                        if (rc != SQLITE.OK)
                        {
                            Debug.Assert(rc == SQLITE.NOMEM);
                            return rc;
                        }
                    }
                    else
                    {
                        if (pPager.eState != PAGER.WRITER_DBMOD)
                            pPg.flags |= PgHdr.PGHDR.NEED_SYNC;
                        PAGERTRACE("APPEND %d page %d needSync=%d\n",
                        PAGERID(pPager), pPg.pgno, (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0);
                    }
                }
                // If the statement journal is open and the page is not in it, then write the current page to the statement journal.  Note that
                // the statement journal format differs from the standard journal format in that it omits the checksums and the header.
                if (subjRequiresPage(pPg))
                    rc = subjournalPage(pPg);
            }
            // Update the database size and return.
            if (pPager.dbSize < pPg.pgno)
                pPager.dbSize = pPg.pgno;
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
            Debug.Assert(assert_pager_state(pPager));
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
                for (var ii = 0; ii < nPage && rc == SQLITE_OK; ii++)
                {
                    var pg = (Pgno)(pg1 + ii);
                    var pPage = new PgHdr();
                    if (pg == pPg.pgno || sqlite3BitvecTest(pPager.pInJournal, pg) == 0)
                    {
                        if (pg != ((PENDING_BYTE / (pPager.pageSize)) + 1))
                        {
                            rc = sqlite3PagerGet(pPager, pg, ref pPage);
                            if (rc == SQLITE.OK)
                            {
                                rc = pager_write(pPage);
                                if ((pPage.flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                                    needSync = true;
                                sqlite3PagerUnref(pPage);
                            }
                        }
                    }
                    else if ((pPage = pager_lookup(pPager, pg)) != null)
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
                        var pPage = pager_lookup(pPager, (Pgno)(pg1 + ii));
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

#if DEBUG
        internal static bool sqlite3PagerIswriteable(DbPage pPg) { return true; }
#else
        internal static bool sqlite3PagerIswriteable(DbPage pPg) { return (pPg.flags & PgHdr.PGHDR.DIRTY) != 0; }        
#endif

        internal static void sqlite3PagerDontWrite(PgHdr pPg)
        {
            var pPager = pPg.pPager;
            if ((pPg.flags & PgHdr.PGHDR.DIRTY) != 0 && pPager.nSavepoint == 0)
            {
                PAGERTRACE("DONT_WRITE page %d of %d\n", pPg.pgno, PAGERID(pPager));
                IOTRACE("CLEAN %p %d\n", pPager, pPg.pgno);
                pPg.flags |= PgHdr.PGHDR.DONT_WRITE;
                pager_set_pagehash(pPg);
            }
        }

        internal static SQLITE pager_incr_changecounter(Pager pPager, bool isDirectMode)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(pPager));
            // Declare and initialize constant integer 'isDirect'. If the atomic-write optimization is enabled in this build, then isDirect
            // is initialized to the value passed as the isDirectMode parameter to this function. Otherwise, it is always set to zero.
            // The idea is that if the atomic-write optimization is not enabled at compile time, the compiler can omit the tests of
            // 'isDirect' below, as well as the block enclosed in the "if( isDirect )" condition.
#if !SQLITE_ENABLE_ATOMIC_WRITE
            var DIRECT_MODE = false;
            Debug.Assert(!isDirectMode);
            UNUSED_PARAMETER(isDirectMode);
#else
            var DIRECT_MODE = isDirectMode;
#endif
            if (!pPager.changeCountDone && pPager.dbSize > 0)
            {
                PgHdr pPgHdr = null; // Reference to page 1
                Debug.Assert(!pPager.tempFile && pPager.fd.isOpen);
                // Open page 1 of the file for writing.
                rc = sqlite3PagerGet(pPager, 1, ref pPgHdr);
                Debug.Assert(pPgHdr == null || rc == SQLITE.OK);
                // If page one was fetched successfully, and this function is not operating in direct-mode, make page 1 writable.  When not in 
                // direct mode, page 1 is always held in cache and hence the PagerGet() above is always successful - hence the ALWAYS on rc==SQLITE_OK.
                if (!DIRECT_MODE && Check.ALWAYS(rc == SQLITE.OK))
                    rc = sqlite3PagerWrite(pPgHdr);
                if (rc == SQLITE.OK)
                {
                    // Actually do the update of the change counter
                    pager_write_changecounter(pPgHdr);
                    // If running in direct mode, write the contents of page 1 to the file.
                    if (DIRECT_MODE)
                    {
                        byte[] zBuf = null;
                        Debug.Assert(pPager.dbFileSize > 0);
                        if (CODEC2(pPager, pPgHdr.pData, 1, SQLITE_ENCRYPT_WRITE_CTX, ref zBuf))
                            return rc = SQLITE.NOMEM;
                        if (rc == SQLITE.OK)
                            rc = sqlite3OsWrite(pPager.fd, zBuf, pPager.pageSize, 0);
                        if (rc == SQLITE.OK)
                            pPager.changeCountDone = true;
                    }
                    else
                        pPager.changeCountDone = true;
                }
                // Release the page reference.
                sqlite3PagerUnref(pPgHdr);
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerSync(Pager pPager)
        {
            var rc = SQLITE.OK;
            if (!pPager.noSync)
            {
                Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0 == MEMDB
#else
0 == pPager.memDb
#endif
);
                rc = sqlite3OsSync(pPager.fd, pPager.syncFlags);
            }
            else if (pPager.fd.isOpen)
            {
                Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0 == MEMDB
#else
0 == pPager.memDb
#endif
);
                sqlite3OsFileControl(pPager.fd, SQLITE_FCNTL_SYNC_OMITTED, ref rc);
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerExclusiveLock(Pager pPager)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD || pPager.eState == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(pPager));
            if (!pagerUseWal(pPager))
                rc = pager_wait_on_lock(pPager, LOCK.EXCLUSIVE);
            return rc;
        }


        internal static SQLITE sqlite3PagerCommitPhaseOne(Pager pPager, string zMaster, bool noSync)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED || pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD || pPager.eState == PAGER.ERROR);
            Debug.Assert(assert_pager_state(pPager));
            // If a prior error occurred, report that error again.
            if (Check.NEVER(pPager.errCode != 0))
                return pPager.errCode;
            PAGERTRACE("DATABASE SYNC: File=%s zMaster=%s nSize=%d\n",
            pPager.zFilename, zMaster, pPager.dbSize);
            // If no database changes have been made, return early.
            if (pPager.eState < PAGER.WRITER_CACHEMOD)
                return SQLITE.OK;
            if (
#if SQLITE_OMIT_MEMORYDB
0 != MEMDB
#else
0 != pPager.memDb
#endif
)
                // If this is an in-memory db, or no pages have been written to, or this function has already been called, it is mostly a no-op.  However, any
                // backup in progress needs to be restarted.
                sqlite3BackupRestart(pPager.pBackup);
            else
            {
                if (pagerUseWal(pPager))
                {
                    var pList = sqlite3PcacheDirtyList(pPager.pPCache);
                    PgHdr pPageOne = null;
                    if (pList == null)
                    {
                        // Must have at least one page for the WAL commit flag.
                        rc = sqlite3PagerGet(pPager, 1, ref pPageOne);
                        pList = pPageOne;
                        pList.pDirty = null;
                    }
                    Debug.Assert(rc == SQLITE.OK);
                    if (Check.ALWAYS(pList))
                        rc = pagerWalFrames(pPager, pList, pPager.dbSize, 1, pPager.fullSync ? pPager.syncFlags : (byte)0);
                    sqlite3PagerUnref(pPageOne);
                    if (rc == SQLITE.OK)
                        sqlite3PcacheCleanAll(pPager.pPCache);
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
                    Debug.Assert(pPager.jfd.isOpen || pPager.journalMode == JOURNALMODE.OFF || pPager.journalMode == JOURNALMODE.WAL);
                    if (!zMaster && pPager.jfd.isOpen
                        && pPager.journalOff == jrnlBufferSize(pPager)
                        && pPager.dbSize >= pPager.dbOrigSize
                        && (0 == (pPg = sqlite3PcacheDirtyList(pPager.pPCache)) || 0 == pPg.pDirty))
                        // Update the db file change counter via the direct-write method. The following call will modify the in-memory representation of page 1
                        // to include the updated change counter and then write page 1 directly to the database file. Because of the atomic-write
                        // property of the host file-system, this is safe.
                        rc = pager_incr_changecounter(pPager, 1);
                    else
                    {
                        rc = sqlite3JournalCreate(pPager.jfd);
                        if (rc == SQLITE.OK)
                            rc = pager_incr_changecounter(pPager, 0);
                    }
#else
                    rc = pager_incr_changecounter(pPager, false);
#endif
                    if (rc != SQLITE.OK)
                        goto commit_phase_one_exit;
                    // If this transaction has made the database smaller, then all pages being discarded by the truncation must be written to the journal
                    // file. This can only happen in auto-vacuum mode.
                    // Before reading the pages with page numbers larger than the current value of Pager.dbSize, set dbSize back to the value
                    // that it took at the start of the transaction. Otherwise, the calls to sqlite3PagerGet() return zeroed pages instead of
                    // reading data from the database file.
#if !SQLITE_OMIT_AUTOVACUUM
                    if (pPager.dbSize < pPager.dbOrigSize && pPager.journalMode != JOURNALMODE.OFF)
                    {
                        Pgno iSkip = PAGER_MJ_PGNO(pPager); // Pending lock page
                        Pgno dbSize = pPager.dbSize;       // Database image size
                        pPager.dbSize = pPager.dbOrigSize;
                        for (Pgno i = dbSize + 1; i <= pPager.dbOrigSize; i++)
                            if (0 == sqlite3BitvecTest(pPager.pInJournal, i) && i != iSkip)
                            {
                                PgHdr pPage = null;             // Page to journal
                                rc = sqlite3PagerGet(pPager, i, ref pPage);
                                if (rc != SQLITE.OK)
                                    goto commit_phase_one_exit;
                                rc = sqlite3PagerWrite(pPage);
                                sqlite3PagerUnref(pPage);
                                if (rc != SQLITE.OK)
                                    goto commit_phase_one_exit;
                            }
                        pPager.dbSize = dbSize;
                    }
#endif

                    // Write the master journal name into the journal file. If a master journal file name has already been written to the journal file,
                    // or if zMaster is NULL (no master journal), then this call is a no-op.
                    rc = writeMasterJournal(pPager, zMaster);
                    if (rc != SQLITE.OK)
                        goto commit_phase_one_exit;
                    // Sync the journal file and write all dirty pages to the database. If the atomic-update optimization is being used, this sync will not 
                    // create the journal file or perform any real IO.
                    // Because the change-counter page was just modified, unless the atomic-update optimization is used it is almost certain that the
                    // journal requires a sync here. However, in locking_mode=exclusive on a system under memory pressure it is just possible that this is 
                    // not the case. In this case it is likely enough that the redundant xSync() call will be changed to a no-op by the OS anyhow. 
                    rc = syncJournal(pPager, 0);
                    if (rc != SQLITE.OK)
                        goto commit_phase_one_exit;
                    rc = pager_write_pagelist(pPager, sqlite3PcacheDirtyList(pPager.pPCache));
                    if (rc != SQLITE.OK)
                    {
                        Debug.Assert(rc != SQLITE.IOERR_BLOCKED);
                        goto commit_phase_one_exit;
                    }
                    sqlite3PcacheCleanAll(pPager.pPCache);
                    // If the file on disk is not the same size as the database image, then use pager_truncate to grow or shrink the file here.
                    if (pPager.dbSize != pPager.dbFileSize)
                    {
                        var nNew = (Pgno)(pPager.dbSize - (pPager.dbSize == PAGER_MJ_PGNO(pPager) ? 1 : 0));
                        Debug.Assert(pPager.eState >= PAGER_WRITER_DBMOD);
                        rc = pager_truncate(pPager, nNew);
                        if (rc != SQLITE.OK)
                            goto commit_phase_one_exit;
                    }
                    // Finally, sync the database file.
                    if (!noSync)
                        rc = sqlite3PagerSync(pPager);
                    IOTRACE("DBSYNC %p\n", pPager);
                }
            }
        commit_phase_one_exit:
            if (rc == SQLITE.OK && !pagerUseWal(pPager))
                pPager.eState = PAGER.WRITER_FINISHED;
            return rc;
        }

        internal static SQLITE sqlite3PagerCommitPhaseTwo(Pager pPager)
        {
            var rc = SQLITE.OK;
            // This routine should not be called if a prior error has occurred. But if (due to a coding error elsewhere in the system) it does get
            // called, just return the same error code without doing anything. */
            if (Check.NEVER(pPager.errCode) != 0)
                return pPager.errCode;
            Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED || pPager.eState == PAGER.WRITER_FINISHED || (pagerUseWal(pPager) && pPager.eState == PAGER.WRITER_CACHEMOD));
            Debug.Assert(assert_pager_state(pPager));
            // An optimization. If the database was not actually modified during this transaction, the pager is running in exclusive-mode and is
            // using persistent journals, then this function is a no-op.
            // The start of the journal file currently contains a single journal header with the nRec field set to 0. If such a journal is used as
            // a hot-journal during hot-journal rollback, 0 changes will be made to the database file. So there is no need to zero the journal
            // header. Since the pager is in exclusive mode, there is no need to drop any locks either.
            if (pPager.eState == PAGER.WRITER_LOCKED && pPager.exclusiveMode && pPager.journalMode == PAGER.JOURNALMODE_PERSIST)
            {
                Debug.Assert(pPager.journalOff == JOURNAL_HDR_SZ(pPager) || 0 == pPager.journalOff);
                pPager.eState = PAGER.READER;
                return SQLITE.OK;
            }
            PAGERTRACE("COMMIT %d\n", PAGERID(pPager));
            rc = pager_end_transaction(pPager, pPager.setMaster);
            return pager_error(pPager, rc);
        }

        internal static int sqlite3PagerRollback(Pager pPager)
        {
            var rc = SQLITE.OK;
            PAGERTRACE("ROLLBACK %d\n", PAGERID(pPager));
            // PagerRollback() is a no-op if called in READER or OPEN state. If the pager is already in the ERROR state, the rollback is not 
            // attempted here. Instead, the error code is returned to the caller.
            Debug.Assert(assert_pager_state(pPager));
            if (pPager.eState == PAGER.ERROR)
                return pPager.errCode;
            if (pPager.eState <= PAGER.READER)
                return SQLITE.OK;
            if (pagerUseWal(pPager))
            {
                rc = sqlite3PagerSavepoint(pPager, SAVEPOINT_ROLLBACK, -1);
                var rc2 = pager_end_transaction(pPager, pPager.setMaster);
                if (rc == SQLITE.OK)
                    rc = rc2;
                rc = pager_error(pPager, rc);
            }
            else if (!pPager.jfd.isOpen || pPager.eState == PAGER.WRITER_LOCKED)
            {
                var eState = pPager.eState;
                rc = pager_end_transaction(pPager, 0);
                if (
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
 && eState > PAGER.WRITER_LOCKED)
                {
                    // This can happen using journal_mode=off. Move the pager to the error  state to indicate that the contents of the cache may not be trusted.
                    // Any active readers will get SQLITE_ABORT.
                    pPager.errCode = SQLITE.ABORT;
                    pPager.eState = PAGER.ERROR;
                    return rc;
                }
            }
            else
                rc = pager_playback(pPager, 0);
            Debug.Assert(pPager.eState == PAGER.READER || rc != SQLITE.OK);
            Debug.Assert(rc == SQLITE.OK || rc == SQLITE.FULL || (rc & 0xFF) == SQLITE.IOERR);
            // If an error occurs during a ROLLBACK, we can no longer trust the pager cache. So call pager_error() on the way out to make any error persistent.
            return pager_error(pPager, rc);
        }

        internal static bool sqlite3PagerIsreadonly(Pager pPager) { return pPager.readOnly; }

        internal static int sqlite3PagerRefcount(Pager pPager) { return sqlite3PcacheRefCount(pPager.pPCache); }

        internal static int sqlite3PagerMemUsed(Pager pPager) { var perPageSize = pPager.pageSize + pPager.nExtra + 20; return perPageSize * sqlite3PcachePagecount(pPager.pPCache) + 0 + pPager.pageSize; }

        internal static int sqlite3PagerPageRefcount(DbPage pPage) { return sqlite3PcachePageRefcount(pPage); }

        internal static bool sqlite3PagerIsMemdb(Pager pPager)
        {
#if SQLITE_OMIT_MEMORYDB
            return MEMDB != 0;
#else
            return pPager.memDb != 0;
#endif
        }

        internal static int sqlite3PagerOpenSavepoint(Pager pPager, int nSavepoint)
        {
            var rc = SQLITE.OK;
            var nCurrent = pPager.nSavepoint;        // Current number of savepoints 
            Debug.Assert(pPager.eState >= PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(pPager));
            if (nSavepoint > nCurrent && pPager.useJournal != 0)
            {
                // Grow the Pager.aSavepoint array using realloc(). Return SQLITE_NOMEM if the allocation fails. Otherwise, zero the new portion in case a 
                // malloc failure occurs while populating it in the for(...) loop below.
                Array.Resize(ref pPager.aSavepoint, nSavepoint);
                var aNew = pPager.aSavepoint; // New Pager.aSavepoint array
                // Populate the PagerSavepoint structures just allocated.
                for (var ii = nCurrent; ii < nSavepoint; ii++)
                {
                    aNew[ii] = new PagerSavepoint();
                    aNew[ii].nOrig = pPager.dbSize;
                    aNew[ii].iOffset = (isOpen(pPager.jfd) && pPager.journalOff > 0 ? pPager.journalOff : (int)JOURNAL_HDR_SZ(pPager));
                    aNew[ii].iSubRec = pPager.nSubRec;
                    aNew[ii].pInSavepoint = sqlite3BitvecCreate(pPager.dbSize);
                    if (pagerUseWal(pPager))
                        sqlite3WalSavepoint(pPager.pWal, aNew[ii].aWalData);
                    pPager.nSavepoint = ii + 1;
                }
                Debug.Assert(pPager.nSavepoint == nSavepoint);
                assertTruncateConstraint(pPager);
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerSavepoint(Pager pPager, int op, int iSavepoint)
        {
            var rc = pPager.errCode;
            Debug.Assert(op == SAVEPOINT_RELEASE || op == SAVEPOINT_ROLLBACK);
            Debug.Assert(iSavepoint >= 0 || op == SAVEPOINT_ROLLBACK);
            if (rc == SQLITE.OK && iSavepoint < pPager.nSavepoint)
            {
                // Figure out how many savepoints will still be active after this operation. Store this value in nNew. Then free resources associated
                // with any savepoints that are destroyed by this operation.
                var nNew = iSavepoint + ((op == SAVEPOINT_RELEASE) ? 0 : 1); // Number of remaining savepoints after this op.
                for (var ii = nNew; ii < pPager.nSavepoint; ii++)
                    sqlite3BitvecDestroy(ref pPager.aSavepoint[ii].pInSavepoint);
                pPager.nSavepoint = nNew;
                // If this is a release of the outermost savepoint, truncate the sub-journal to zero bytes in size.
                if (op == SAVEPOINT_RELEASE)
                    if (nNew == 0 && isOpen(pPager.sjfd))
                    {
                        // Only truncate if it is an in-memory sub-journal.
                        if (sqlite3IsMemJournal(pPager.sjfd))
                        {
                            rc = sqlite3OsTruncate(pPager.sjfd, 0);
                            Debug.Assert(rc == SQLITE_OK);
                        }
                        pPager.nSubRec = 0;
                    }
                    // Else this is a rollback operation, playback the specified savepoint. If this is a temp-file, it is possible that the journal file has
                    // not yet been opened. In this case there have been no changes to the database file, so the playback operation can be skipped.
                    else if (pagerUseWal(pPager) || pPager.jfd.isOpen)
                    {
                        var pSavepoint = (nNew == 0 ? (PagerSavepoint)null : pPager.aSavepoint[nNew - 1]);
                        rc = pagerPlaybackSavepoint(pPager, pSavepoint);
                        Debug.Assert(rc != SQLITE.DONE);
                    }
            }
            return rc;
        }

        internal static string sqlite3PagerFilename(Pager pPager) { return pPager.zFilename; }
        internal static sqlite3_vfs sqlite3PagerVfs(Pager pPager) { return pPager.pVfs; }
        internal static sqlite3_file sqlite3PagerFile(Pager pPager) { return pPager.fd; }
        internal static string sqlite3PagerJournalname(Pager pPager) { return pPager.zJournal; }
        internal static bool sqlite3PagerNosync(Pager pPager) { return pPager.noSync; }

#if SQLITE_HAS_CODEC
        internal static void sqlite3PagerSetCodec(Pager pPager, dxCodec xCodec, dxCodecSizeChng xCodecSizeChng, dxCodecFree xCodecFree, codec_ctx pCodec)
        {
            if (pPager.xCodecFree != null)
                pPager.xCodecFree(ref pPager.pCodec);
            pPager.xCodec = (pPager.memDb != 0 ? null : xCodec);
            pPager.xCodecSizeChng = xCodecSizeChng;
            pPager.xCodecFree = xCodecFree;
            pPager.pCodec = pCodec;
            pagerReportSize(pPager);
        }

        internal static object sqlite3PagerGetCodec(Pager pPager) { return pPager.pCodec; }
#endif

#if !SQLITE_OMIT_AUTOVACUUM
        internal static SQLITE sqlite3PagerMovepage(Pager pPager, DbPage pPg, Pgno pgno, int isCommit)
        {
            Debug.Assert(pPg.nRef > 0);
            Debug.Assert(pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(pPager));
            // In order to be able to rollback, an in-memory database must journal the page we are moving from.
            SQLITE rc;
            if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
pPager.memDb != 0
#endif
)
            {
                rc = sqlite3PagerWrite(pPg);
                if (rc != 0)
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
            PAGERTRACE("MOVE %d page %d (needSync=%d) moves to %d\n",
            PAGERID(pPager), pPg.pgno, (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0, pgno);
            IOTRACE("MOVE %p %d %d\n", pPager, pPg.pgno, pgno);
            // If the journal needs to be sync()ed before page pPg.pgno can be written to, store pPg.pgno in local variable needSyncPgno.
            // If the isCommit flag is set, there is no need to remember that the journal needs to be sync()ed before database page pPg.pgno
            // can be written to. The caller has already promised not to write to it.
            if (((pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0) && 0 == isCommit)
            {
                needSyncPgno = pPg.pgno;
                Debug.Assert(pageInJournal(pPg) || pPg.pgno > pPager.dbOrigSize);
                Debug.Assert((pPg.flags & PgHdr.PGHDR.DIRTY) != 0);
            }
            // If the cache contains a page with page-number pgno, remove it from its hash chain. Also, if the PGHDR_NEED_SYNC was set for
            // page pgno before the 'move' operation, it needs to be retained for the page moved there.
            pPg.flags &= ~PgHdr.PGHDR.NEED_SYNC;
            pPgOld = pager_lookup(pPager, pgno);
            Debug.Assert(null == pPgOld || pPgOld.nRef == 1);
            if (pPgOld != null)
            {
                pPg.flags |= (pPgOld.flags & PgHdr.PGHDR.NEED_SYNC);
                if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
pPager.memDb != 0
#endif
)
                    // Do not discard pages from an in-memory database since we might need to rollback later.  Just move the page out of the way.
                    sqlite3PcacheMove(pPgOld, pPager.dbSize + 1);
                else
                    sqlite3PcacheDrop(pPgOld);
            }
            origPgno = pPg.pgno;
            sqlite3PcacheMove(pPg, pgno);
            sqlite3PcacheMakeDirty(pPg);
            // For an in-memory database, make sure the original page continues to exist, in case the transaction needs to roll back.  Use pPgOld
            // as the original page since it has already been allocated.
            if (
#if SQLITE_OMIT_MEMORYDB
0!=MEMDB
#else
0 != pPager.memDb
#endif
)
            {
                Debug.Assert(pPgOld);
                sqlite3PcacheMove(pPgOld, origPgno);
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
                rc = sqlite3PagerGet(pPager, needSyncPgno, ref pPgHdr);
                if (rc != SQLITE.OK)
                {
                    if (needSyncPgno <= pPager.dbOrigSize)
                    {
                        Debug.Assert(pPager.pTmpSpace != null);
                        var pTemp = new uint[pPager.pTmpSpace.Length];
                        sqlite3BitvecClear(pPager.pInJournal, needSyncPgno, pTemp);
                    }
                    return rc;
                }
                pPgHdr.flags |= PgHdr.PGHDR.NEED_SYNC;
                sqlite3PcacheMakeDirty(pPgHdr);
                sqlite3PagerUnref(pPgHdr);
            }
            return SQLITE.OK;
        }
#endif

        internal static byte[] sqlite3PagerGetData(DbPage pPg) { Debug.Assert(pPg.nRef > 0 || pPg.pPager.memDb != 0); return pPg.pData; }
        internal static MemPage sqlite3PagerGetExtra(DbPage pPg) { return pPg.pExtra; }
        internal static bool sqlite3PagerLockingMode(Pager pPager, int eMode)
        {
            Debug.Assert(eMode == LOCKINGMODE.QUERY || eMode == LOCKINGMODE.NORMAL || eMode == LOCKINGMODE.EXCLUSIVE);
            Debug.Assert(LOCKINGMODE.QUERY < 0);
            Debug.Assert(LOCKINGMODE.NORMAL >= 0 && LOCKINGMODE.EXCLUSIVE >= 0);
            Debug.Assert(pPager.exclusiveMode || !sqlite3WalHeapMemory(pPager.pWal));
            if (eMode >= 0 && !pPager.tempFile && !sqlite3WalHeapMemory(pPager.pWal))
                pPager.exclusiveMode = eMode != 0;
            return pPager.exclusiveMode;
        }

        internal static JOURNALMODE sqlite3PagerSetJournalMode(Pager pPager, JOURNALMODE eMode)
        {
            var eOld = pPager.journalMode;    // Prior journalmode
#if DEBUG
            // The print_pager_state() routine is intended to be used by the debugger only.  We invoke it once here to suppress a compiler warning. */
            print_pager_state(pPager);
#endif
            // The eMode parameter is always valid
            Debug.Assert(eMode == JOURNALMODE.DELETE
                || eMode == JOURNALMODE.TRUNCATE
                || eMode == JOURNALMODE.PERSIST
                || eMode == JOURNALMODE.OFF
                || eMode == JOURNALMODE.WAL
                || eMode == JOURNALMODE.MEMORY);
            // This routine is only called from the OP_JournalMode opcode, and the logic there will never allow a temporary file to be changed to WAL mode.
            Debug.Assert(pPager.tempFile == false || eMode != JOURNALMODE.WAL);
            // Do allow the journalmode of an in-memory database to be set to anything other than MEMORY or OFF
            if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
1 == pPager.memDb
#endif
)
            {
                Debug.Assert(eOld == JOURNALMODE.MEMORY || eOld == JOURNALMODE.OFF);
                if (eMode != JOURNALMODE.MEMORY && eMode != JOURNALMODE.OFF)
                    eMode = eOld;
            }
            if (eMode != eOld)
            {
                // Change the journal mode.
                Debug.Assert(pPager.eState != PAGER.ERROR);
                pPager.journalMode = (byte)eMode;
                // When transistioning from TRUNCATE or PERSIST to any other journal mode except WAL, unless the pager is in locking_mode=exclusive mode,
                // delete the journal file.
                Debug.Assert((JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert((JOURNALMODE.PERSIST & 5) == 1);
                Debug.Assert((JOURNALMODE.DELETE & 5) == 0);
                Debug.Assert((JOURNALMODE.MEMORY & 5) == 4);
                Debug.Assert((JOURNALMODE.OFF & 5) == 0);
                Debug.Assert((JOURNALMODE.WAL & 5) == 5);
                Debug.Assert(pPager.fd.isOpen || pPager.exclusiveMode);
                if (!pPager.exclusiveMode && (eOld & 5) == 1 && (eMode & 1) == 0)
                {
                    // In this case we would like to delete the journal file. If it is not possible, then that is not a problem. Deleting the journal file
                    // here is an optimization only.
                    // Before deleting the journal file, obtain a RESERVED lock on the database file. This ensures that the journal file is not deleted
                    // while it is in use by some other client.
                    sqlite3OsClose(pPager.jfd);
                    if (pPager.eLock >= LOCK.RESERVED)
                        sqlite3OsDelete(pPager.pVfs, pPager.zJournal, 0);
                    else
                    {
                        var rc = SQLITE.OK;
                        var state = pPager.eState;
                        Debug.Assert(state == PAGER.OPEN || state == PAGER.READER);
                        if (state == PAGER.OPEN)
                            rc = sqlite3PagerSharedLock(pPager);
                        if (pPager.eState == PAGER.READER)
                        {
                            Debug.Assert(rc == SQLITE.OK);
                            rc = pagerLockDb(pPager, LOCK.RESERVED);
                        }
                        if (rc == SQLITE.OK)
                            sqlite3OsDelete(pPager.pVfs, pPager.zJournal, 0);
                        if (rc == SQLITE.OK && state == PAGER.READER)
                            pagerUnlockDb(pPager, LOCK.SHARED);
                        else if (state == PAGER.OPEN)
                            pager_unlock(pPager);
                        Debug.Assert(state == pPager.eState);
                    }
                }
            }
            // Return the new journal mode
            return pPager.journalMode;
        }

        internal static JOURNALMODE sqlite3PagerGetJournalMode(Pager pPager) { return pPager.journalMode; }

        internal static bool sqlite3PagerOkToChangeJournalMode(Pager pPager)
        {
            Debug.Assert(assert_pager_state(pPager));
            if (pPager.eState >= PAGER.WRITER_CACHEMOD)
                return false;
            if (Check.NEVER(pPager.jfd.isOpen && pPager.journalOff > 0))
                return false;
            return true;
        }

        internal static long sqlite3PagerJournalSizeLimit(Pager pPager, long iLimit)
        {
            if (iLimit >= -1) { pPager.journalSizeLimit = iLimit; sqlite3WalLimit(pPager.pWal, iLimit); }
            return pPager.journalSizeLimit;
        }

        internal static sqlite3_backup sqlite3PagerBackupPtr(Pager pPager) { return pPager.pBackup; }

































    }
}
