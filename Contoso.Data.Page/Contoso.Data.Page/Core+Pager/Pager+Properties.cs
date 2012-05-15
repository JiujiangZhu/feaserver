using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using IPCache = Contoso.Core.Name.PCache1;
using Pgno = System.UInt32;
using VFSLOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {
#if SQLITE_HAS_CODEC
        internal void pagerReportSize()
        {
            if (this.xCodecSizeChng != null)
                this.xCodecSizeChng(this.pCodec, this.pageSize, this.nReserve);
        }
#else
        internal  void pagerReportSize( ) { }
#endif

        internal void setSectorSize()
        {
            Debug.Assert(this.fd.isOpen || this.tempFile);
            if (!this.tempFile)
            {
                // Sector size doesn't matter for temporary files. Also, the file may not have been opened yet, in which case the OsSectorSize()
                // call will segfault.
                this.sectorSize = (Pgno)FileEx.sqlite3OsSectorSize(this.fd);
            }
            if (this.sectorSize < 32)
            {
                Debug.Assert(MAX_SECTOR_SIZE >= 512);
                this.sectorSize = 512;
            }
            if (this.sectorSize > MAX_SECTOR_SIZE)
                this.sectorSize = MAX_SECTOR_SIZE;
        }

        internal void sqlite3PagerSetCachesize(int mxPage) { this.pPCache.sqlite3PcacheSetCachesize(mxPage); }

#if !SQLITE_OMIT_PAGER_PRAGMAS
        internal void sqlite3PagerSetSafetyLevel(int level, int bFullFsync, int bCkptFullFsync)
        {
            Debug.Assert(level >= 1 && level <= 3);
            this.noSync = (level == 1 || this.tempFile);
            this.fullSync = (level == 3 && !this.tempFile);
            if (this.noSync)
            {
                this.syncFlags = 0;
                this.ckptSyncFlags = 0;
            }
            else if (bFullFsync != 0)
            {
                this.syncFlags = VirtualFile.SYNC.FULL;
                this.ckptSyncFlags = VirtualFile.SYNC.FULL;
            }
            else if (bCkptFullFsync != 0)
            {
                this.syncFlags = VirtualFile.SYNC.NORMAL;
                this.ckptSyncFlags = VirtualFile.SYNC.FULL;
            }
            else
            {
                this.syncFlags = VirtualFile.SYNC.NORMAL;
                this.ckptSyncFlags = VirtualFile.SYNC.NORMAL;
            }
        }
#endif

        internal void sqlite3PagerSetBusyhandler(Func<object, int> xBusyHandler, object pBusyHandlerArg)
        {
            this.xBusyHandler = xBusyHandler;
            this.pBusyHandlerArg = pBusyHandlerArg;
        }

        internal SQLITE sqlite3PagerSetPagesize(ref uint pPageSize, int nReserve)
        {
            // It is not possible to do a full assert_pager_state() here, as this function may be called from within PagerOpen(), before the state
            // of the Pager object is internally consistent.
            // At one point this function returned an error if the pager was in PAGER_ERROR state. But since PAGER_ERROR state guarantees that
            // there is at least one outstanding page reference, this function is a no-op for that case anyhow.
            var rc = SQLITE.OK;
            var pageSize = pPageSize;
            Debug.Assert(pageSize == 0 || (pageSize >= 512 && pageSize <= SQLITE_MAX_PAGE_SIZE));
            if ((this.memDb == 0 || this.dbSize == 0) && this.pPCache.sqlite3PcacheRefCount() == 0 && pageSize != 0 && pageSize != (uint)this.pageSize)
            {
                long nByte = 0;
                if (this.eState > PAGER.OPEN && this.fd.isOpen)
                    rc = FileEx.sqlite3OsFileSize(this.fd, ref nByte);
                if (rc == SQLITE.OK)
                {
                    pager_reset();
                    this.dbSize = (Pgno)(nByte / pageSize);
                    this.pageSize = (int)pageSize;
                    IPCache.sqlite3PageFree(ref this.pTmpSpace);
                    this.pTmpSpace = MallocEx.sqlite3Malloc((int)pageSize);
                    this.pPCache.sqlite3PcacheSetPageSize((int)pageSize);
                }
            }
            pPageSize = (uint)this.pageSize;
            if (rc == SQLITE.OK)
            {
                if (nReserve < 0)
                    nReserve = this.nReserve;
                Debug.Assert(nReserve >= 0 && nReserve < 1000);
                this.nReserve = (short)nReserve;
                pagerReportSize();
            }
            return rc;
        }

        internal byte[] sqlite3PagerTempSpace()
        {
            return this.pTmpSpace;
        }

        internal Pgno sqlite3PagerMaxPageCount(int mxPage)
        {
            if (mxPage > 0)
                this.mxPgno = (Pgno)mxPage;
            Debug.Assert(this.eState != PAGER.OPEN);     // Called only by OP_MaxPgcnt
            Debug.Assert(this.mxPgno >= this.dbSize);  // OP_MaxPgcnt enforces this
            return this.mxPgno;
        }

        internal void sqlite3PagerPagecount(out Pgno pnPage)
        {
            Debug.Assert(this.eState >= PAGER.READER);
            Debug.Assert(this.eState != PAGER.WRITER_FINISHED);
            pnPage = this.dbSize;
        }

        internal static Pgno sqlite3PagerPagenumber(DbPage pPg) { return pPg.pgno; }
        internal static void sqlite3PagerRef(DbPage pPg) { PCache.sqlite3PcacheRef(pPg); }
#if DEBUG
        internal static bool sqlite3PagerIswriteable(DbPage pPg) { return true; }
#else
        internal static bool sqlite3PagerIswriteable(DbPage pPg) { return (pPg.flags & PgHdr.PGHDR.DIRTY) != 0; }        
#endif
        internal bool sqlite3PagerIsreadonly() { return this.readOnly; }
        internal int sqlite3PagerRefcount() { return this.pPCache.sqlite3PcacheRefCount(); }
        internal int sqlite3PagerMemUsed() { var perPageSize = this.pageSize + this.nExtra + 20; return perPageSize * this.pPCache.sqlite3PcachePagecount() + 0 + this.pageSize; }
        internal static int sqlite3PagerPageRefcount(DbPage pPage) { return PCache.sqlite3PcachePageRefcount(pPage); }
        internal bool sqlite3PagerIsMemdb()
        {
#if SQLITE_OMIT_MEMORYDB
            return MEMDB != 0;
#else
            return this.memDb != 0;
#endif
        }


        internal string sqlite3PagerFilename() { return this.zFilename; }
        internal VirtualFileSystem sqlite3PagerVfs() { return this.pVfs; }
        internal VirtualFile sqlite3PagerFile() { return this.fd; }
        internal string sqlite3PagerJournalname() { return this.zJournal; }
        internal bool sqlite3PagerNosync() { return this.noSync; }

#if SQLITE_HAS_CODEC
        internal void sqlite3PagerSetCodec(codec_ctx.dxCodec xCodec, codec_ctx.dxCodecSizeChng xCodecSizeChng, codec_ctx.dxCodecFree xCodecFree, codec_ctx pCodec)
        {
            if (this.xCodecFree != null)
                this.xCodecFree(ref this.pCodec);
            this.xCodec = (this.memDb != 0 ? null : xCodec);
            this.xCodecSizeChng = xCodecSizeChng;
            this.xCodecFree = xCodecFree;
            this.pCodec = pCodec;
            pagerReportSize();
        }

        internal object sqlite3PagerGetCodec() { return this.pCodec; }
#endif


        internal static byte[] sqlite3PagerGetData(DbPage pPg) { Debug.Assert(pPg.nRef > 0 || pPg.pPager.memDb != 0); return pPg.pData; }
        internal static MemPage sqlite3PagerGetExtra(DbPage pPg) { return pPg.pExtra; }
        internal bool sqlite3PagerLockingMode(LOCKINGMODE eMode)
        {
            Debug.Assert(eMode == LOCKINGMODE.QUERY || eMode == LOCKINGMODE.NORMAL || eMode == LOCKINGMODE.EXCLUSIVE);
            Debug.Assert(LOCKINGMODE.QUERY < 0);
            Debug.Assert(LOCKINGMODE.NORMAL >= 0 && LOCKINGMODE.EXCLUSIVE >= 0);
            Debug.Assert(this.exclusiveMode || !this.pWal.sqlite3WalHeapMemory());
            if (eMode >= 0 && !this.tempFile && !this.pWal.sqlite3WalHeapMemory())
                this.exclusiveMode = eMode != 0;
            return this.exclusiveMode;
        }

        internal JOURNALMODE sqlite3PagerSetJournalMode(JOURNALMODE eMode)
        {
            var eOld = this.journalMode;    // Prior journalmode
#if DEBUG
            // The print_pager_state() routine is intended to be used by the debugger only.  We invoke it once here to suppress a compiler warning. */
            this.print_pager_state();
#endif
            // The eMode parameter is always valid
            Debug.Assert(eMode == JOURNALMODE.DELETE
                || eMode == JOURNALMODE.TRUNCATE
                || eMode == JOURNALMODE.PERSIST
                || eMode == JOURNALMODE.OFF
                || eMode == JOURNALMODE.WAL
                || eMode == JOURNALMODE.MEMORY);
            // This routine is only called from the OP_JournalMode opcode, and the logic there will never allow a temporary file to be changed to WAL mode.
            Debug.Assert(this.tempFile == false || eMode != JOURNALMODE.WAL);
            // Do allow the journalmode of an in-memory database to be set to anything other than MEMORY or OFF
            if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
1 == this.memDb
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
                Debug.Assert(this.eState != PAGER.ERROR);
                this.journalMode = eMode;
                // When transistioning from TRUNCATE or PERSIST to any other journal mode except WAL, unless the pager is in locking_mode=exclusive mode,
                // delete the journal file.
                Debug.Assert(((int)JOURNALMODE.TRUNCATE & 5) == 1);
                Debug.Assert(((int)JOURNALMODE.PERSIST & 5) == 1);
                Debug.Assert(((int)JOURNALMODE.DELETE & 5) == 0);
                Debug.Assert(((int)JOURNALMODE.MEMORY & 5) == 4);
                Debug.Assert(((int)JOURNALMODE.OFF & 5) == 0);
                Debug.Assert(((int)JOURNALMODE.WAL & 5) == 5);
                Debug.Assert(this.fd.isOpen || this.exclusiveMode);
                if (!this.exclusiveMode && ((int)eOld & 5) == 1 && ((int)eMode & 1) == 0)
                {
                    // In this case we would like to delete the journal file. If it is not possible, then that is not a problem. Deleting the journal file
                    // here is an optimization only.
                    // Before deleting the journal file, obtain a RESERVED lock on the database file. This ensures that the journal file is not deleted
                    // while it is in use by some other client.
                    FileEx.sqlite3OsClose(this.jfd);
                    if (this.eLock >= VFSLOCK.RESERVED)
                        FileEx.sqlite3OsDelete(this.pVfs, this.zJournal, 0);
                    else
                    {
                        var rc = SQLITE.OK;
                        var state = this.eState;
                        Debug.Assert(state == PAGER.OPEN || state == PAGER.READER);
                        if (state == PAGER.OPEN)
                            rc = this.sqlite3PagerSharedLock();
                        if (this.eState == PAGER.READER)
                        {
                            Debug.Assert(rc == SQLITE.OK);
                            rc = this.pagerLockDb(VFSLOCK.RESERVED);
                        }
                        if (rc == SQLITE.OK)
                            FileEx.sqlite3OsDelete(this.pVfs, this.zJournal, 0);
                        if (rc == SQLITE.OK && state == PAGER.READER)
                            this.pagerUnlockDb(VFSLOCK.SHARED);
                        else if (state == PAGER.OPEN)
                            this.pager_unlock();
                        Debug.Assert(state == this.eState);
                    }
                }
            }
            // Return the new journal mode
            return this.journalMode;
        }

        internal JOURNALMODE sqlite3PagerGetJournalMode() { return this.journalMode; }

        internal bool sqlite3PagerOkToChangeJournalMode()
        {
            Debug.Assert(this.assert_pager_state());
            if (this.eState >= PAGER.WRITER_CACHEMOD)
                return false;
            if (Check.NEVER(this.jfd.isOpen && this.journalOff > 0))
                return false;
            return true;
        }

        internal long sqlite3PagerJournalSizeLimit(long iLimit)
        {
            if (iLimit >= -1) { this.journalSizeLimit = iLimit; this.pWal.sqlite3WalLimit(iLimit); }
            return this.journalSizeLimit;
        }

        internal IBackup sqlite3PagerBackupPtr() { return this.pBackup; }
    }
}
