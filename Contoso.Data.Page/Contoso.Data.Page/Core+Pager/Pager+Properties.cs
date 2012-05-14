using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using IPCache = Contoso.Core.Name.PCache1;
using LOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {

#if SQLITE_HAS_CODEC
        internal static void pagerReportSize(Pager pPager)
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

        internal static void setSectorSize(Pager pPager)
        {
            Debug.Assert(pPager.fd.isOpen || pPager.tempFile);
            if (!pPager.tempFile)
            {
                // Sector size doesn't matter for temporary files. Also, the file may not have been opened yet, in which case the OsSectorSize()
                // call will segfault.
                pPager.sectorSize = (Pgno)FileEx.sqlite3OsSectorSize(pPager.fd);
            }
            if (pPager.sectorSize < 32)
            {
                Debug.Assert(MAX_SECTOR_SIZE >= 512);
                pPager.sectorSize = 512;
            }
            if (pPager.sectorSize > MAX_SECTOR_SIZE)
                pPager.sectorSize = MAX_SECTOR_SIZE;
        }

        internal static void sqlite3PagerSetCachesize(Pager pPager, int mxPage) { pPager.pPCache.sqlite3PcacheSetCachesize(mxPage); }

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
                pPager.syncFlags = VirtualFile.SYNC.FULL;
                pPager.ckptSyncFlags = VirtualFile.SYNC.FULL;
            }
            else if (bCkptFullFsync != 0)
            {
                pPager.syncFlags = VirtualFile.SYNC.NORMAL;
                pPager.ckptSyncFlags = VirtualFile.SYNC.FULL;
            }
            else
            {
                pPager.syncFlags = VirtualFile.SYNC.NORMAL;
                pPager.ckptSyncFlags = VirtualFile.SYNC.NORMAL;
            }
        }
#endif

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
            if ((pPager.memDb == 0 || pPager.dbSize == 0) && pPager.pPCache.sqlite3PcacheRefCount() == 0 && pageSize != 0 && pageSize != (uint)pPager.pageSize)
            {
                long nByte = 0;
                if (pPager.eState > PAGER.OPEN && pPager.fd.isOpen)
                    rc = FileEx.sqlite3OsFileSize(pPager.fd, ref nByte);
                if (rc == SQLITE.OK)
                {
                    pager_reset(pPager);
                    pPager.dbSize = (Pgno)(nByte / pageSize);
                    pPager.pageSize = (int)pageSize;
                    IPCache.sqlite3PageFree(ref pPager.pTmpSpace);
                    pPager.pTmpSpace = MallocEx.sqlite3Malloc((int)pageSize);
                    pPager.pPCache.sqlite3PcacheSetPageSize((int)pageSize);
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

        internal static void sqlite3PagerPagecount(Pager pPager, out Pgno pnPage)
        {
            Debug.Assert(pPager.eState >= PAGER.READER);
            Debug.Assert(pPager.eState != PAGER.WRITER_FINISHED);
            pnPage = pPager.dbSize;
        }

        internal static Pgno sqlite3PagerPagenumber(DbPage pPg) { return pPg.pgno; }

        internal static void sqlite3PagerRef(DbPage pPg) { PCache.sqlite3PcacheRef(pPg); }


#if DEBUG
        internal static bool sqlite3PagerIswriteable(DbPage pPg) { return true; }
#else
        internal static bool sqlite3PagerIswriteable(DbPage pPg) { return (pPg.flags & PgHdr.PGHDR.DIRTY) != 0; }        
#endif

        internal static bool sqlite3PagerIsreadonly(Pager pPager) { return pPager.readOnly; }

        internal static int sqlite3PagerRefcount(Pager pPager) { return pPager.pPCache.sqlite3PcacheRefCount(); }

        internal static int sqlite3PagerMemUsed(Pager pPager) { var perPageSize = pPager.pageSize + pPager.nExtra + 20; return perPageSize * pPager.pPCache.sqlite3PcachePagecount() + 0 + pPager.pageSize; }

        internal static int sqlite3PagerPageRefcount(DbPage pPage) { return PCache.sqlite3PcachePageRefcount(pPage); }

        internal static bool sqlite3PagerIsMemdb(Pager pPager)
        {
#if SQLITE_OMIT_MEMORYDB
            return MEMDB != 0;
#else
            return pPager.memDb != 0;
#endif
        }


        internal static string sqlite3PagerFilename(Pager pPager) { return pPager.zFilename; }
        internal static VirtualFileSystem sqlite3PagerVfs(Pager pPager) { return pPager.pVfs; }
        internal static VirtualFile sqlite3PagerFile(Pager pPager) { return pPager.fd; }
        internal static string sqlite3PagerJournalname(Pager pPager) { return pPager.zJournal; }
        internal static bool sqlite3PagerNosync(Pager pPager) { return pPager.noSync; }

#if SQLITE_HAS_CODEC
        internal static void sqlite3PagerSetCodec(Pager pPager, codec_ctx.dxCodec xCodec, codec_ctx.dxCodecSizeChng xCodecSizeChng, codec_ctx.dxCodecFree xCodecFree, codec_ctx pCodec)
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


        internal static byte[] sqlite3PagerGetData(DbPage pPg) { Debug.Assert(pPg.nRef > 0 || pPg.pPager.memDb != 0); return pPg.pData; }
        internal static MemPage sqlite3PagerGetExtra(DbPage pPg) { return pPg.pExtra; }
        internal static bool sqlite3PagerLockingMode(Pager pPager, LOCKINGMODE eMode)
        {
            Debug.Assert(eMode == LOCKINGMODE.QUERY || eMode == LOCKINGMODE.NORMAL || eMode == LOCKINGMODE.EXCLUSIVE);
            Debug.Assert(LOCKINGMODE.QUERY < 0);
            Debug.Assert(LOCKINGMODE.NORMAL >= 0 && LOCKINGMODE.EXCLUSIVE >= 0);
            Debug.Assert(pPager.exclusiveMode || !Wal.sqlite3WalHeapMemory(pPager.pWal));
            if (eMode >= 0 && !pPager.tempFile && !Wal.sqlite3WalHeapMemory(pPager.pWal))
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
                pPager.journalMode = eMode;
                // When transistioning from TRUNCATE or PERSIST to any other journal mode except WAL, unless the pager is in locking_mode=exclusive mode,
                // delete the journal file.
                Debug.Assert(((int)JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert(((int)JOURNALMODE.PERSIST & 5) == 1);
                Debug.Assert(((int)JOURNALMODE.DELETE & 5) == 0);
                Debug.Assert(((int)JOURNALMODE.MEMORY & 5) == 4);
                Debug.Assert(((int)JOURNALMODE.OFF & 5) == 0);
                Debug.Assert(((int)JOURNALMODE.WAL & 5) == 5);
                Debug.Assert(pPager.fd.isOpen || pPager.exclusiveMode);
                if (!pPager.exclusiveMode && ((int)eOld & 5) == 1 && ((int)eMode & 1) == 0)
                {
                    // In this case we would like to delete the journal file. If it is not possible, then that is not a problem. Deleting the journal file
                    // here is an optimization only.
                    // Before deleting the journal file, obtain a RESERVED lock on the database file. This ensures that the journal file is not deleted
                    // while it is in use by some other client.
                    FileEx.sqlite3OsClose(pPager.jfd);
                    if (pPager.eLock >= LOCK.RESERVED)
                        FileEx.sqlite3OsDelete(pPager.pVfs, pPager.zJournal, 0);
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
                            FileEx.sqlite3OsDelete(pPager.pVfs, pPager.zJournal, 0);
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
            if (iLimit >= -1) { pPager.journalSizeLimit = iLimit; Wal.sqlite3WalLimit(pPager.pWal, iLimit); }
            return pPager.journalSizeLimit;
        }

        internal static IBackup sqlite3PagerBackupPtr(Pager pPager) { return pPager.pBackup; }
    }
}
