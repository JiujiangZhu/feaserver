using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using System.Text;
using OPEN = Contoso.Sys.VirtualFileSystem.OPEN;
using LOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal static SQLITE pager_open_journal(Pager pPager)
        {
            var rc = SQLITE.OK;
            var pVfs = pPager.pVfs;
            Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(pPager));
            Debug.Assert(pPager.pInJournal == null);

            // If already in the error state, this function is a no-op.  But on the other hand, this routine is never called if we are already in
            // an error state. */
            if (Check.NEVER(pPager.errCode))
                return pPager.errCode;
            if (!pPager.pagerUseWal() && pPager.journalMode != JOURNALMODE.OFF)
            {
                pPager.pInJournal = Bitvec.sqlite3BitvecCreate(pPager.dbSize);
                // Open the journal file if it is not already open.
                if (!pPager.jfd.isOpen)
                {
                    if (pPager.journalMode == JOURNALMODE.MEMORY)
                        pPager.jfd = new MemJournalFile();
                    else
                    {
                        var flags = OPEN.READWRITE | OPEN.CREATE | (pPager.tempFile ? OPEN.DELETEONCLOSE | OPEN.TEMP_JOURNAL : OPEN.MAIN_JOURNAL);
#if SQLITE_ENABLE_ATOMIC_WRITE
                        rc = sqlite3JournalOpen(pVfs, pPager.zJournal, pPager.jfd, flags, jrnlBufferSize(pPager));
#else
                        OPEN int0 = 0;
                        rc = FileEx.sqlite3OsOpen(pVfs, pPager.zJournal, pPager.jfd, flags, ref int0);
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
                Bitvec.sqlite3BitvecDestroy(ref pPager.pInJournal);
                pPager.pInJournal = null;
            }
            else
            {
                Debug.Assert(pPager.eState == PAGER.WRITER_LOCKED);
                pPager.eState = PAGER.WRITER_CACHEMOD;
            }
            return rc;
        }

        internal static PgHdr pager_lookup(Pager pPager, Pgno pgno)
        {
            // It is not possible for a call to PcacheFetch() with createFlag==0 to fail, since no attempt to allocate dynamic memory will be made.
            PgHdr p = null;
            pPager.pPCache.sqlite3PcacheFetch(pgno, 0, ref p);
            return p;
        }

        internal static void pager_reset(Pager pPager)
        {
            pPager.pBackup.sqlite3BackupRestart();
            pPager.pPCache.sqlite3PcacheClear();
        }

        internal static void pager_unlock(Pager pPager)
        {
            Debug.Assert(pPager.eState == PAGER.READER
                || pPager.eState == PAGER.OPEN
                || pPager.eState == PAGER.ERROR);
            Bitvec.sqlite3BitvecDestroy(ref pPager.pInJournal);
            pPager.pInJournal = null;
            releaseAllSavepoints(pPager);
            if (pagerUseWal(pPager))
            {
                Debug.Assert(!pPager.jfd.isOpen);
                Wal.sqlite3WalEndReadTransaction(pPager.pWal);
                pPager.eState = PAGER.OPEN;
            }
            else if (!pPager.exclusiveMode)
            {
                var iDc = (pPager.fd.isOpen ? FileEx.sqlite3OsDeviceCharacteristics(pPager.fd) : 0);
                // If the operating system support deletion of open files, then close the journal file when dropping the database lock.  Otherwise
                // another connection with journal_mode=delete might delete the file out from under us.
                Debug.Assert(((int)JOURNALMODE.MEMORY & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.OFF & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.WAL & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.DELETE & 5) != 1);
                Debug.Assert(((int)JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert(((int)JOURNALMODE.PERSIST & 5) == 1);
                if (0 == (iDc & VirtualFile.IOCAP.UNDELETABLE_WHEN_OPEN) || 1 != ((int)pPager.journalMode & 5))
                    FileEx.sqlite3OsClose(pPager.jfd);
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
                if (pPager.jfd is MemJournalFile)
                {
                    Debug.Assert(pPager.journalMode == JOURNALMODE.MEMORY);
                    FileEx.sqlite3OsClose(pPager.jfd);
                }
                else if (pPager.journalMode == JOURNALMODE.TRUNCATE)
                {
                    rc = (pPager.journalOff == 0 ? SQLITE.OK : FileEx.sqlite3OsTruncate(pPager.jfd, 0));
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
                    FileEx.sqlite3OsClose(pPager.jfd);
                    if (!pPager.tempFile)
                        rc = FileEx.sqlite3OsDelete(pPager.pVfs, pPager.zJournal, 0);
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
            Bitvec.sqlite3BitvecDestroy(ref pPager.pInJournal);
            pPager.pInJournal = null;
            pPager.nRec = 0;
            pPager.pPCache.sqlite3PcacheCleanAll();
            pPager.pPCache.sqlite3PcacheTruncate(pPager.dbSize);
            var rc2 = SQLITE.OK;    // Error code from db file unlock operation
            if (pagerUseWal(pPager))
            {
                // Drop the WAL write-lock, if any. Also, if the connection was in locking_mode=exclusive mode but is no longer, drop the EXCLUSIVE 
                // lock held on the database file.
                rc2 = Wal.sqlite3WalEndWriteTransaction(pPager.pWal);
                Debug.Assert(rc2 == SQLITE.OK);
            }
            if (!pPager.exclusiveMode && (!pagerUseWal(pPager) || Wal.sqlite3WalExclusiveMode(pPager.pWal, 0)))
            {
                rc2 = pagerUnlockDb(pPager, LOCK.SHARED);
                pPager.changeCountDone = false;
            }
            pPager.eState = PAGER.READER;
            pPager.setMaster = 0;
            return (rc == SQLITE.OK ? rc2 : rc);
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
            rc = FileEx.sqlite3OsRead(jfd, aData, pPager.pageSize, pOffset + 4);
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
            if (pgno > pPager.dbSize || pDone.sqlite3BitvecTest(pgno) != 0)
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
            if (pDone != null && (rc = pDone.sqlite3BitvecSet(pgno)) != SQLITE.OK)
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
                rc = FileEx.sqlite3OsWrite(pPager.fd, aData, pPager.pageSize, ofst);
                if (pgno > pPager.dbFileSize)
                    pPager.dbFileSize = pgno;
                if (pPager.pBackup != null)
                {
                    if (CODEC1(pPager, aData, pgno, codec_ctx.DECRYPT))
                        rc = SQLITE.NOMEM;
                    pPager.pBackup.sqlite3BackupUpdate(pgno, aData);
                    if (CODEC2(pPager, aData, pgno, codec_ctx.ENCRYPT_READ_CTX, ref aData))
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
                rc = pPager.sqlite3PagerAcquire(pgno, ref pPg, 1);
                Debug.Assert(pPager.doNotSpill == 1);
                pPager.doNotSpill--;
                if (rc != SQLITE.OK)
                    return rc;
                pPg.flags &= ~PgHdr.PGHDR.NEED_READ;
                PCache.sqlite3PcacheMakeDirty(pPg);
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
                    PCache.sqlite3PcacheMakeClean(pPg);
                }
                pager_set_pagehash(pPg);
                // If this was page 1, then restore the value of Pager.dbFileVers. Do this before any decoding.
                if (pgno == 1)
                    Buffer.BlockCopy(pData, 24, pPager.dbFileVers, 0, pPager.dbFileVers.Length);
                // Decode the page just read from disk
                if (CODEC1(pPager, pData, pPg.pgno, codec_ctx.DECRYPT))
                    rc = SQLITE.NOMEM;
                PCache.sqlite3PcacheRelease(pPg);
            }
            return rc;
        }

        internal static SQLITE pager_delmaster(Pager pPager, string zMaster)
        {
            //string zMasterJournal = null; /* Contents of master journal file */
            //long nMasterJournal;           /* Size of master journal file */
            //string zJournal;              /* Pointer to one journal within MJ file */
            //string zMasterPtr;            /* Space to hold MJ filename from a journal file */
            //int nMasterPtr;               /* Amount of space allocated to zMasterPtr[] */
            // Allocate space for both the pJournal and pMaster file descriptors. If successful, open the master journal file for reading.
            var pMaster = new VirtualFile();// Malloc'd master-journal file descriptor
            var pJournal = new VirtualFile();// Malloc'd child-journal file descriptor
            var pVfs = pPager.pVfs;
            OPEN iDummy = 0;
            var rc = FileEx.sqlite3OsOpen(pVfs, zMaster, pMaster, OPEN.READONLY | OPEN.MASTER_JOURNAL, ref iDummy);
            if (rc != SQLITE.OK)
                goto delmaster_out;
            Debugger.Break();
            //TODO --
            // Load the entire master journal file into space obtained from sqlite3_malloc() and pointed to by zMasterJournal.   Also obtain
            // sufficient space (in zMasterPtr) to hold the names of master journal files extracted from regular rollback-journals.
            //rc = sqlite3OsFileSize(pMaster, &nMasterJournal);
            //if (rc != SQLITE.OK) goto delmaster_out;
            //nMasterPtr = pVfs.mxPathname + 1;
            //  zMasterJournal = sqlite3Malloc((int)nMasterJournal + nMasterPtr + 1);
            //  if ( !zMasterJournal )
            //  {
            //    rc = SQLITE_NOMEM;
            //    goto delmaster_out;
            //  }
            //  zMasterPtr = &zMasterJournal[nMasterJournal+1];
            //  rc = sqlite3OsRead( pMaster, zMasterJournal, (int)nMasterJournal, 0 );
            //  if ( rc != SQLITE.OK ) goto delmaster_out;
            //  zMasterJournal[nMasterJournal] = 0;
            //  zJournal = zMasterJournal;
            //  while ( ( zJournal - zMasterJournal ) < nMasterJournal )
            //  {
            //    int exists;
            //    rc = sqlite3OsAccess( pVfs, zJournal, SQLITE_ACCESS_EXISTS, &exists );
            //    if ( rc != SQLITE.OK )
            //      goto delmaster_out;
            //    if ( exists )
            //    {
            //      // One of the journals pointed to by the master journal exists. Open it and check if it points at the master journal. If
            //      // so, return without deleting the master journal file.
            //      int c;
            //      int flags = ( SQLITE_OPEN_READONLY | SQLITE_OPEN_MAIN_JOURNAL );
            //      rc = sqlite3OsOpen( pVfs, zJournal, pJournal, flags, 0 );
            //      if ( rc != SQLITE.OK )
            //        goto delmaster_out;
            //      rc = readMasterJournal( pJournal, zMasterPtr, nMasterPtr );
            //      sqlite3OsClose( pJournal );
            //      if ( rc != SQLITE.OK )
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
                FileEx.sqlite3OsClose(pMaster);
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
                rc = FileEx.sqlite3OsFileSize(pPager.fd, ref currentSize);
                var newSize = szPage * nPage;
                if (rc == SQLITE.OK && currentSize != newSize)
                {
                    if (currentSize > newSize)
                        rc = FileEx.sqlite3OsTruncate(pPager.fd, newSize);
                    else
                    {
                        var pTmp = pPager.pTmpSpace;
                        Array.Clear(pTmp, 0, szPage);
                        rc = FileEx.sqlite3OsWrite(pPager.fd, pTmp, szPage, newSize - szPage);
                    }
                    if (rc == SQLITE.OK)
                        pPager.dbSize = nPage;
                }
            }
            return rc;
        }

        internal static SQLITE pager_playback(Pager pPager, int isHot)
        {
            var pVfs = pPager.pVfs;
            // Figure out how many records are in the journal.  Abort early if the journal is empty.
            Debug.Assert(pPager.jfd.isOpen);
            long szJ = 0;            // Size of the journal file in bytes
            var res = 1;             // Value returned by sqlite3OsAccess()
            var zMaster = new byte[pPager.pVfs.mxPathname + 1]; // Name of master journal file if any
            var rc = FileEx.sqlite3OsFileSize(pPager.jfd, ref szJ);
            if (rc != SQLITE.OK)
                goto end_playback;
            // Read the master journal name from the journal, if it is present. If a master journal file name is specified, but the file is not
            // present on disk, then the journal is not hot and does not need to be played back.
            // TODO: Technically the following is an error because it assumes that buffer Pager.pTmpSpace is (mxPathname+1) bytes or larger. i.e. that
            // (pPager.pageSize >= pPager.pVfs.mxPathname+1). Using os_unix.c, mxPathname is 512, which is the same as the minimum allowable value
            // for pageSize.
            rc = readMasterJournal(pPager.jfd, zMaster, (uint)pPager.pVfs.mxPathname + 1);
            if (rc == SQLITE.OK && zMaster[0] != 0)
                rc = FileEx.sqlite3OsAccess(pVfs, Encoding.UTF8.GetString(zMaster, 0, zMaster.Length), VirtualFileSystem.ACCESS.EXISTS, ref res);
            zMaster = null;
            if (rc != SQLITE.OK || res == 0)
                goto end_playback;
            pPager.journalOff = 0;
            // This loop terminates either when a readJournalHdr() or pager_playback_one_page() call returns SQLITE_DONE or an IO error occurs.
            var needPagerReset = isHot; // True to reset page prior to first page rollback
            Pgno mxPg = 0;            // Size of the original file in pages
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
                        else if (rc == SQLITE.IOERR_SHORT_READ)
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
            Debug.Assert(!pPager.fd.isOpen || FileEx.sqlite3OsFileControl(pPager.fd, VirtualFile.FCNTL.DB_UNCHANGED, ref iDummy) >= SQLITE.OK);
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

        internal static void pager_write_changecounter(PgHdr pPg)
        {
            // Increment the value just read and write it back to byte 24.
            uint change_counter = ConvertEx.sqlite3Get4byte(pPg.pPager.dbFileVers, 0) + 1;
            ConvertEx.put32bits(pPg.pData, 24, change_counter);
            // Also store the SQLite version number in bytes 96..99 and in bytes 92..95 store the change counter for which the version number
            // is valid.
            ConvertEx.put32bits(pPg.pData, 92, change_counter);
            ConvertEx.put32bits(pPg.pData, 96, SysEx.SQLITE_VERSION_NUMBER);
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

        internal static SQLITE pager_write_pagelist(Pager pPager, PgHdr pList)
        {
            var rc = SQLITE.OK;
            // This function is only called for rollback pagers in WRITER_DBMOD state. 
            Debug.Assert(!pagerUseWal(pPager));
            Debug.Assert(pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(pPager.eLock == LOCK.EXCLUSIVE);
            // If the file is a temp-file has not yet been opened, open it now. It is not possible for rc to be other than SQLITE.OK if this branch
            // is taken, as pager_wait_on_lock() is a no-op for temp-files.
            if (!pPager.fd.isOpen)
            {
                Debug.Assert(pPager.tempFile && rc == SQLITE.OK);
                rc = pagerOpentemp(pPager, ref pPager.fd, pPager.vfsFlags);
            }
            // Before the first write, give the VFS a hint of what the final file size will be.
            Debug.Assert(rc != SQLITE.OK || pPager.fd.isOpen);
            if (rc == SQLITE.OK && pPager.dbSize > pPager.dbHintSize)
            {
                long szFile = pPager.pageSize * (long)pPager.dbSize;
                FileEx.sqlite3OsFileControl(pPager.fd, VirtualFile.FCNTL.SIZE_HINT, ref szFile);
                pPager.dbHintSize = pPager.dbSize;
            }
            while (rc == SQLITE.OK && pList)
            {
                var pgno = pList.pgno;
                // If there are dirty pages in the page cache with page numbers greater than Pager.dbSize, this means sqlite3PagerTruncateImage() was called to
                // make the file smaller (presumably by auto-vacuum code). Do not write any such pages to the file.
                // Also, do not write out any page that has the PGHDR_DONT_WRITE flag set (set by sqlite3PagerDontWrite()).
                if (pList.pgno <= pPager.dbSize && 0 == (pList.flags & PgHdr.PGHDR.DONT_WRITE))
                {
                    Debug.Assert((pList.flags & PgHdr.PGHDR.NEED_SYNC) == 0);
                    if (pList.pgno == 1)
                        pager_write_changecounter(pList);
                    // Encode the database
                    byte[] pData = null; // Data to write
                    if (CODEC2(pPager, pList.pData, pgno, codec_ctx.ENCRYPT_WRITE_CTX, ref pData))
                        return SQLITE.NOMEM;
                    // Write out the page data.
                    long offset = (pList.pgno - 1) * (long)pPager.pageSize;   // Offset to write
                    rc = FileEx.sqlite3OsWrite(pPager.fd, pData, pPager.pageSize, offset);
                    // If page 1 was just written, update Pager.dbFileVers to match the value now stored in the database file. If writing this
                    // page caused the database file to grow, update dbFileSize.
                    if (pgno == 1)
                        Buffer.BlockCopy(pData, 24, pPager.dbFileVers, 0, pPager.dbFileVers.Length);
                    if (pgno > pPager.dbFileSize)
                        pPager.dbFileSize = pgno;
                    // Update any backup objects copying the contents of this pager.
                    pPager.pBackup.sqlite3BackupUpdate(pgno, pList.pData);
                    PAGERTRACE("STORE %d page %d hash(%08x)\n",
                    PAGERID(pPager), pgno, pager_pagehash(pList));
                    SysEx.IOTRACE("PGOUT %p %d\n", pPager, pgno);
                }
                else
                    PAGERTRACE("NOSTORE %d page %d\n", PAGERID(pPager), pgno);
                pager_set_pagehash(pList);
                pList = pList.pDirty;
            }
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
            if (Check.NEVER(pPager.errCode))
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
            PCache.sqlite3PcacheMakeDirty(pPg);
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
                        Debug.Assert(pPg.pgno != ((VirtualFile.PENDING_BYTE / (pPager.pageSize)) + 1));
                        Debug.Assert(pPager.journalHdr <= pPager.journalOff);
                        byte[] pData2 = null;
                        if (CODEC2(pPager, pData, pPg.pgno, codec_ctx.ENCRYPT_READ_CTX, ref pData2))
                            return SQLITE.NOMEM;
                        var cksum = pager_cksum(pPager, pData2);
                        // Even if an IO or diskfull error occurred while journalling the page in the block above, set the need-sync flag for the page.
                        // Otherwise, when the transaction is rolled back, the logic in playback_one_page() will think that the page needs to be restored
                        // in the database file. And if an IO error occurs while doing so, then corruption may follow.
                        pPg.flags |= PgHdr.PGHDR.NEED_SYNC;
                        rc = write32bits(pPager.jfd, iOff, pPg.pgno);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = FileEx.sqlite3OsWrite(pPager.jfd, pData2, pPager.pageSize, iOff + 4);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = write32bits(pPager.jfd, iOff + pPager.pageSize + 4, cksum);
                        if (rc != SQLITE.OK)
                            return rc;
                        SysEx.IOTRACE("JOUT %p %d %lld %d\n", pPager, pPg.pgno,
                        pPager.journalOff, pPager.pageSize);
                        PAGERTRACE("JOURNAL %d page %d needSync=%d hash(%08x)\n",
                        PAGERID(pPager), pPg.pgno, (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0 ? 1 : 0, pager_pagehash(pPg));
                        pPager.journalOff += 8 + pPager.pageSize;
                        pPager.nRec++;
                        Debug.Assert(pPager.pInJournal != null);
                        rc = pPager.pInJournal.sqlite3BitvecSet(pPg.pgno);
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
            SysEx.UNUSED_PARAMETER(isDirectMode);
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
                // direct mode, page 1 is always held in cache and hence the PagerGet() above is always successful - hence the ALWAYS on rc==SQLITE.OK.
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
                        if (CODEC2(pPager, pPgHdr.pData, 1, codec_ctx.ENCRYPT_WRITE_CTX, ref zBuf))
                            return rc = SQLITE.NOMEM;
                        if (rc == SQLITE.OK)
                            rc = FileEx.sqlite3OsWrite(pPager.fd, zBuf, pPager.pageSize, 0);
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
    }
}
