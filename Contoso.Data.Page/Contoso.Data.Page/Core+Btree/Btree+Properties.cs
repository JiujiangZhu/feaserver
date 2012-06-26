using System;
using System.Diagnostics;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;
using SAVEPOINT = Contoso.Core.Pager.SAVEPOINT;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;

namespace Contoso.Core
{
    public partial class Btree
    {
        // was:sqlite3BtreePager
        public Pager GetPager()
        {
            return this.Shared.Pager;
        }

        // was:sqlite3BtreeGetFilename
        public string GetFilename()
        {
            Debug.Assert(this.Shared.Pager != null);
            return this.Shared.Pager.sqlite3PagerFilename();
        }

        // was:sqlite3BtreeGetJournalname
        public string GetJournalName()
        {
            Debug.Assert(this.Shared.Pager != null);
            return this.Shared.Pager.sqlite3PagerJournalname();
        }

        // was:sqlite3BtreeIsInTrans
        public bool IsInTrans()
        {
            Debug.Assert(this == null || MutexEx.Held(this.DB.Mutex));
            return (this != null && (this.inTrans == TRANS.WRITE));
        }

        // was:sqlite3BtreeLastPage
        public Pgno GetLastPage()
        {
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(((this.Shared.Pages) & 0x8000000) == 0);
            return (Pgno)this.Shared.btreePagecount();
        }

        // was:sqlite3BtreeSetCacheSize
        public RC SetCacheSize(int mxPage)
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            pBt.Pager.sqlite3PagerSetCachesize(mxPage);
            sqlite3BtreeLeave();
            return RC.OK;
        }

#if !SQLITE_OMIT_PAGER_PRAGMAS
        // was:sqlite3BtreeSetSafetyLevel
        public RC SetSafetyLevel(int level, int fullSync, int ckptFullSync)
        {
            var pBt = this.Shared;
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            Debug.Assert(level >= 1 && level <= 3);
            sqlite3BtreeEnter();
            pBt.Pager.sqlite3PagerSetSafetyLevel(level, fullSync, ckptFullSync);
            sqlite3BtreeLeave();
            return RC.OK;
        }
#endif

        // was:sqlite3BtreeSetPageSize
        public RC SetPageSize(int pageSize, int nReserve, int iFix)
        {
            Debug.Assert(nReserve >= -1 && nReserve <= 255);
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            if (pBt.PageSizeFixed)
            {
                sqlite3BtreeLeave();
                return RC.READONLY;
            }
            if (nReserve < 0)
                nReserve = (int)(pBt.PageSize - pBt.UsableSize);
            Debug.Assert(nReserve >= 0 && nReserve <= 255);
            if (pageSize >= 512 && pageSize <= Pager.SQLITE_MAX_PAGE_SIZE && ((pageSize - 1) & pageSize) == 0)
            {
                Debug.Assert((pageSize & 7) == 0);
                Debug.Assert(pBt.Page1 == null && pBt.Cursors == null);
                pBt.PageSize = (uint)pageSize;
            }
            var rc = pBt.Pager.sqlite3PagerSetPagesize(ref pBt.PageSize, nReserve);
            pBt.UsableSize = (ushort)(pBt.PageSize - nReserve);
            if (iFix != 0)
                pBt.PageSizeFixed = true;
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeGetPageSize
        public int GetPageSize() { return (int)this.Shared.PageSize; }

        // was:sqlite3BtreeSetAutoVacuum
        public RC SetAutoVacuum(int autoVacuum)
        {
#if SQLITE_OMIT_AUTOVACUUM
            return SQLITE.READONLY;
#else
            sqlite3BtreeEnter();
            var rc = RC.OK;
            var av = (byte)autoVacuum;
            var pBt = this.Shared;
            if (pBt.PageSizeFixed && (av != 0) != pBt.AutoVacuum)
                rc = RC.READONLY;
            else
            {
                pBt.AutoVacuum = av != 0;
                pBt.IncrVacuum = av == 2;
            }
            sqlite3BtreeLeave();
            return rc;
#endif
        }

        // was:sqlite3BtreeGetAutoVacuum
        public AUTOVACUUM GetAutoVacuum()
        {
#if SQLITE_OMIT_AUTOVACUUM
return BTREE.AUTOVACUUM_NONE;
#else
            sqlite3BtreeEnter();
            var rc = (!this.Shared.AutoVacuum ? AUTOVACUUM.NONE : (!this.Shared.IncrVacuum ? AUTOVACUUM.FULL : AUTOVACUUM.INCR));
            sqlite3BtreeLeave();
            return rc;
#endif
        }

#if !SQLITE_OMIT_PAGER_PRAGMAS || !SQLITE_OMIT_VACUUM
        // was:sqlite3BtreeGetReserve
        public int GetReserve()
        {
            sqlite3BtreeEnter();
            var n = (int)(this.Shared.PageSize - this.Shared.UsableSize);
            sqlite3BtreeLeave();
            return n;
        }

        // was:sqlite3BtreeMaxPageCount
        public Pgno GetMaxPageCount(int mxPage)
        {
            sqlite3BtreeEnter();
            var n = (Pgno)this.Shared.Pager.sqlite3PagerMaxPageCount(mxPage);
            sqlite3BtreeLeave();
            return n;
        }

        // was:sqlite3BtreeSecureDelete
        public int SecureDelete(int newFlag)
        {
            if (this == null)
                return 0;
            sqlite3BtreeEnter();
            if (newFlag >= 0)
                this.Shared.SecureDelete = (newFlag != 0);
            var b = (this.Shared.SecureDelete ? 1 : 0);
            sqlite3BtreeLeave();
            return b;
        }
#endif

        // was:sqlite3BtreeIsInReadTrans
        public bool IsInReadTrans()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            return (this.inTrans != TRANS.NONE);
        }

        // was:sqlite3BtreeIsInBackup
        public bool IsInBackup()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            return (this.nBackup != 0);
        }

        // was:sqlite3BtreeSetVersion
        public RC SetVersion(int iVersion)
        {
            var pBt = this.Shared;
            Debug.Assert(this.inTrans == TRANS.NONE);
            Debug.Assert(iVersion == 1 || iVersion == 2);
            // If setting the version fields to 1, do not automatically open the WAL connection, even if the version fields are currently set to 2.
            pBt.DoNotUseWal = iVersion == 1;
            var rc = BeginTrans(0);
            if (rc == RC.OK)
            {
                var aData = pBt.Page1.Data;
                if (aData[18] != (byte)iVersion || aData[19] != (byte)iVersion)
                {
                    rc = BeginTrans(2);
                    if (rc == RC.OK)
                    {
                        rc = Pager.sqlite3PagerWrite(pBt.Page1.DbPage);
                        if (rc == RC.OK)
                        {
                            aData[18] = (byte)iVersion;
                            aData[19] = (byte)iVersion;
                        }
                    }
                }
            }
            pBt.DoNotUseWal = false;
            return rc;
        }
    }
}
