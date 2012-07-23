using System;
using System.Diagnostics;
using Contoso.Core.Name;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using DbPage = Contoso.Core.PgHdr;
using MUTEX = Contoso.Threading.MutexEx.MUTEX;
using Pgno = System.UInt32;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;
using TRANS = Contoso.Core.Btree.TRANS;
using Contoso.Threading;

namespace Contoso.Core
{
    public partial class BtShared
    {
        internal RC saveAllCursors(Pgno iRoot, BtreeCursor pExcept)
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            Debug.Assert(pExcept == null || pExcept.Shared == this);
            for (var p = this.Cursors; p != null; p = p.Next)
                if (p != pExcept && (0 == iRoot || p.RootID == iRoot) && p.State == CURSOR.VALID)
                {
                    var rc = p.SavePosition();
                    if (rc != RC.OK)
                        return rc;
                }
            return RC.OK;
        }

#if !SQLITE_OMIT_AUTOVACUUM
        internal Pgno ptrmapPageno(Pgno pgno)
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            if (pgno < 2)
                return 0;
            var nPagesPerMapPage = (int)(this.UsableSize / 5 + 1);
            var iPtrMap = (Pgno)((pgno - 2) / nPagesPerMapPage);
            var ret = (Pgno)(iPtrMap * nPagesPerMapPage) + 2;
            if (ret == MemPage.PENDING_BYTE_PAGE(this))
                ret++;
            return ret;
        }

        internal void ptrmapPut(Pgno key, PTRMAP eType, Pgno parent, ref RC rRC)
        {
            if (rRC != RC.OK)
                return;
            Debug.Assert(MutexEx.Held(this.Mutex));
            // The master-journal page number must never be used as a pointer map page
            Debug.Assert(!MemPage.PTRMAP_ISPAGE(this, MemPage.PENDING_BYTE_PAGE(this)));
            Debug.Assert(this.AutoVacuum);
            if (key == 0)
            {
                rRC = SysEx.SQLITE_CORRUPT_BKPT();
                return;
            }
            var iPtrmap = MemPage.PTRMAP_PAGENO(this, key);
            var pDbPage = new PgHdr();  // The pointer map page 
            var rc = this.Pager.Get(iPtrmap, ref pDbPage);
            if (rc != RC.OK)
            {
                rRC = rc;
                return;
            }
            var offset = (int)MemPage.PTRMAP_PTROFFSET(iPtrmap, key);
            if (offset < 0)
            {
                rRC = SysEx.SQLITE_CORRUPT_BKPT();
                goto ptrmap_exit;
            }
            Debug.Assert(offset <= (int)this.UsableSize - 5);
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage); // The pointer map data 
            if (eType != (PTRMAP)pPtrmap[offset] || ConvertEx.Get4(pPtrmap, offset + 1) != parent)
            {
                Btree.TRACE("PTRMAP_UPDATE: {0}->({1},{2})", key, eType, parent);
                rRC = rc = Pager.Write(pDbPage);
                if (rc == RC.OK)
                {
                    pPtrmap[offset] = (byte)eType;
                    ConvertEx.Put4L(pPtrmap, offset + 1, parent);
                }
            }
        ptrmap_exit:
            Pager.Unref(pDbPage);
        }

        internal RC ptrmapGet(Pgno key, ref PTRMAP pEType, ref Pgno pPgno)
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            var iPtrmap = (int)MemPage.PTRMAP_PAGENO(this, key);
            var pDbPage = new PgHdr(); // The pointer map page
            var rc = this.Pager.Get((Pgno)iPtrmap, ref pDbPage);
            if (rc != RC.OK)
                return rc;
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage);// Pointer map page data
            var offset = (int)MemPage.PTRMAP_PTROFFSET((Pgno)iPtrmap, key);
            if (offset < 0)
            {
                Pager.Unref(pDbPage);
                return SysEx.SQLITE_CORRUPT_BKPT();
            }
            Debug.Assert(offset <= (int)this.UsableSize - 5);
            var v = pPtrmap[offset];
            if (v < 1 || v > 5)
                return SysEx.SQLITE_CORRUPT_BKPT();
            pEType = (PTRMAP)v;
            pPgno = ConvertEx.Get4(pPtrmap, offset + 1);
            Pager.Unref(pDbPage);
            return RC.OK;
        }

#else
        internal Pgno ptrmapPageno(Pgno pgno) { return 0; }
        internal void ptrmapPut(Pgno key, PTRMAP eType, Pgno parent, ref SQLITE pRC) { pRC = SQLITE.OK; }
        internal SQLITE ptrmapGet(Pgno key, ref PTRMAP pEType, ref Pgno pPgno) { return SQLITE.OK; }
#endif

        internal bool removeFromSharingList()
        {
#if !SQLITE_OMIT_SHARED_CACHE
            sqlite3_mutex pMaster;
            BtShared list;
            bool removed = false;
            Debug.Assert(MutexEx.sqlite3_mutex_notheld(Mutex));
            pMaster = MutexEx.sqlite3MutexAlloc(MUTEX.STATIC_MASTER);
            MutexEx.sqlite3_mutex_enter(pMaster);
            nRef--;
            if (nRef <= 0)
            {
                if (SysEx.getGLOBAL<BtShared>(Btree.s_sqlite3SharedCacheList) == this)
                    SysEx.setGLOBAL<BtShared>(Btree.s_sqlite3SharedCacheList, Next);
                else
                {
                    list = SysEx.getGLOBAL<BtShared>(Btree.s_sqlite3SharedCacheList);
                    while (Check.ALWAYS(list) != null && list.Next != this)
                        list = list.Next;
                    if (Check.ALWAYS(list) != null)
                        list.Next = Next;
                }
                if (MutexEx.SQLITE_THREADSAFE)
                    MutexEx.sqlite3_mutex_free(Mutex);
                removed = true;
            }
            MutexEx.sqlite3_mutex_leave(pMaster);
            return removed;
#else
            return true;
#endif
        }

        internal void allocateTempSpace() { if (this.pTmpSpace == null) this.pTmpSpace = MallocEx.sqlite3Malloc((int)this.PageSize); }
        internal void freeTempSpace() { PCache1.sqlite3PageFree(ref this.pTmpSpace); }

        internal RC lockBtree()
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            Debug.Assert(this.Page1 == null);
            var rc = this.Pager.SharedLock();
            if (rc != RC.OK)
                return rc;
            MemPage pPage1 = null; // Page 1 of the database file
            rc = btreeGetPage(1, ref pPage1, 0);
            if (rc != RC.OK)
                return rc;
            // Do some checking to help insure the file we opened really is a valid database file.
            Pgno nPageHeader;      // Number of pages in the database according to hdr
            var nPage = nPageHeader = ConvertEx.Get4(pPage1.Data, 28); // Number of pages in the database
            Pgno nPageFile;    // Number of pages in the database file
            this.Pager.GetPageCount(out nPageFile);
            if (nPage == 0 || ArrayEx.Compare(pPage1.Data, 24, pPage1.Data, 92, 4) != 0)
                nPage = nPageFile;
            if (nPage > 0)
            {
                var page1 = pPage1.Data;
                rc = RC.NOTADB;
                if (ArrayEx.Compare(page1, Btree.zMagicHeader, 16) != 0)
                    goto page1_init_failed;
#if SQLITE_OMIT_WAL
                if (page1[18] > 1)
                    this.ReadOnly = true;
                if (page1[19] > 1)
                {
                    this.Schema.file_format = page1[19];
                    goto page1_init_failed;
                }
#else
if( page1[18]>2 ){
pBt.readOnly = true;
}
if( page1[19]>2 ){
goto page1_init_failed;
}

/* If the write version is set to 2, this database should be accessed
** in WAL mode. If the log is not already open, open it now. Then 
** return SQLITE_OK and return without populating BtShared.pPage1.
** The caller detects this and calls this function again. This is
** required as the version of page 1 currently in the page1 buffer
** may not be the latest version - there may be a newer one in the log
** file.
*/
if( page1[19]==2 && pBt.doNotUseWAL==false ){
int isOpen = 0;
rc = sqlite3PagerOpenWal(pBt.pPager, ref isOpen);
if( rc!=SQLITE_OK ){
goto page1_init_failed;
}else if( isOpen==0 ){
releasePage(pPage1);
return SQLITE_OK;
}
rc = SQLITE_NOTADB;
}
#endif
                // The maximum embedded fraction must be exactly 25%.  And the minimum embedded fraction must be 12.5% for both leaf-data and non-leaf-data.
                // The original design allowed these amounts to vary, but as of version 3.6.0, we require them to be fixed.
                if (ArrayEx.Compare(page1, 21, "\x0040\x0020\x0020", 3) != 0) // "\100\040\040"
                    goto page1_init_failed;
                var pageSize = (uint)((page1[16] << 8) | (page1[17] << 16));
                if (((pageSize - 1) & pageSize) != 0 || pageSize > Pager.SQLITE_MAX_PAGE_SIZE || pageSize <= 256)
                    goto page1_init_failed;
                Debug.Assert((pageSize & 7) == 0);
                var usableSize = pageSize - page1[20];
                if (pageSize != this.PageSize)
                {
                    // After reading the first page of the database assuming a page size of BtShared.pageSize, we have discovered that the page-size is
                    // actually pageSize. Unlock the database, leave pBt.pPage1 at zero and return SQLITE_OK. The caller will call this function
                    // again with the correct page-size.
                    pPage1.releasePage();
                    this.UsableSize = usableSize;
                    this.PageSize = pageSize;
                    rc = this.Pager.SetPageSize(ref this.PageSize, (int)(pageSize - usableSize));
                    return rc;
                }
                if ((this.DB.flags & sqlite3b.SQLITE.RecoveryMode) == 0 && nPage > nPageFile)
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto page1_init_failed;
                }
                if (usableSize < 480)
                    goto page1_init_failed;
                this.PageSize = pageSize;
                this.UsableSize = usableSize;
#if !SQLITE_OMIT_AUTOVACUUM
                this.AutoVacuum = (ConvertEx.Get4(page1, 36 + 4 * 4) != 0);
                this.IncrVacuum = (ConvertEx.Get4(page1, 36 + 7 * 4) != 0);
#endif
            }
            // maxLocal is the maximum amount of payload to store locally for a cell.  Make sure it is small enough so that at least minFanout
            // cells can will fit on one page.  We assume a 10-byte page header. Besides the payload, the cell must store:
            //     2-byte pointer to the cell
            //     4-byte child pointer
            //     9-byte nKey value
            //     4-byte nData value
            //     4-byte overflow page pointer
            // So a cell consists of a 2-byte pointer, a header which is as much as 17 bytes long, 0 to N bytes of payload, and an optional 4 byte overflow page pointer.
            this.MaxLocal = (ushort)((this.UsableSize - 12) * 64 / 255 - 23);
            this.MinLocal = (ushort)((this.UsableSize - 12) * 32 / 255 - 23);
            this.MaxLeaf = (ushort)(this.UsableSize - 35);
            this.MinLeaf = (ushort)((this.UsableSize - 12) * 32 / 255 - 23);
            Debug.Assert(this.MaxLeaf + 23 <= Btree.MX_CELL_SIZE(this));
            this.Page1 = pPage1;
            this.Pages = nPage;
            return RC.OK;
        page1_init_failed:
            pPage1.releasePage();
            this.Page1 = null;
            return rc;
        }

        internal void unlockBtreeIfUnused()
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            Debug.Assert(this.Cursors == null || this.InTransaction > TRANS.NONE);
            if (this.InTransaction == TRANS.NONE && this.Page1 != null)
            {
                Debug.Assert(this.Page1.Data != null);
                //Debug.Assert(pBt.pPager.sqlite3PagerRefcount() == 1 );
                this.Page1.releasePage();
                this.Page1 = null;
            }
        }

        internal RC newDatabase()
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            if (this.Pages > 0)
                return RC.OK;
            var pP1 = this.Page1;
            Debug.Assert(pP1 != null);
            var data = pP1.Data;
            var rc = Pager.Write(pP1.DbPage);
            if (rc != RC.OK)
                return rc;
            Buffer.BlockCopy(Btree.zMagicHeader, 0, data, 0, 16);
            Debug.Assert(Btree.zMagicHeader.Length == 16);
            data[16] = (byte)((this.PageSize >> 8) & 0xff);
            data[17] = (byte)((this.PageSize >> 16) & 0xff);
            data[18] = 1;
            data[19] = 1;
            Debug.Assert(this.UsableSize <= this.PageSize && this.UsableSize + 255 >= this.PageSize);
            data[20] = (byte)(this.PageSize - this.UsableSize);
            data[21] = 64;
            data[22] = 32;
            data[23] = 32;
            pP1.zeroPage(Btree.PTF_INTKEY | Btree.PTF_LEAF | Btree.PTF_LEAFDATA);
            this.PageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
            Debug.Assert(this.AutoVacuum == true || this.AutoVacuum == false);
            Debug.Assert(this.IncrVacuum == true || this.IncrVacuum == false);
            ConvertEx.Put4(data, 36 + 4 * 4, this.AutoVacuum ? 1 : 0);
            ConvertEx.Put4(data, 36 + 7 * 4, this.IncrVacuum ? 1 : 0);
#endif
            this.Pages = 1;
            data[31] = 1;
            return RC.OK;
        }

        internal RC getOverflowPage(Pgno ovfl, out MemPage ppPage, out Pgno pPgnoNext)
        {
            Pgno next = 0;
            MemPage pPage = null;
            ppPage = null;
            var rc = RC.OK;
            Debug.Assert(MutexEx.Held(this.Mutex));
            // Debug.Assert( pPgnoNext != 0);
#if !SQLITE_OMIT_AUTOVACUUM
            // Try to find the next page in the overflow list using the autovacuum pointer-map pages. Guess that the next page in
            // the overflow list is page number (ovfl+1). If that guess turns out to be wrong, fall back to loading the data of page
            // number ovfl to determine the next page number.
            if (this.AutoVacuum)
            {
                Pgno pgno = 0;
                Pgno iGuess = ovfl + 1;
                PTRMAP eType = 0;
                while (MemPage.PTRMAP_ISPAGE(this, iGuess) || iGuess == MemPage.PENDING_BYTE_PAGE(this))
                    iGuess++;
                if (iGuess <= btreePagecount())
                {
                    rc = ptrmapGet(iGuess, ref eType, ref pgno);
                    if (rc == RC.OK && eType == PTRMAP.OVERFLOW2 && pgno == ovfl)
                    {
                        next = iGuess;
                        rc = RC.DONE;
                    }
                }
            }
#endif
            Debug.Assert(next == 0 || rc == RC.DONE);
            if (rc == RC.OK)
            {
                rc = btreeGetPage(ovfl, ref pPage, 0);
                Debug.Assert(rc == RC.OK || pPage == null);
                if (rc == RC.OK)
                    next = ConvertEx.Get4(pPage.Data);
            }
            pPgnoNext = next;
            if (ppPage != null)
                ppPage = pPage;
            else
                pPage.releasePage();
            return (rc == RC.DONE ? RC.OK : rc);
        }

        internal RC clearDatabasePage(Pgno pgno, int freePageFlag, ref int pnChange)
        {
            var pPage = new MemPage();
            Debug.Assert(MutexEx.Held(this.Mutex));
            if (pgno > btreePagecount())
                return SysEx.SQLITE_CORRUPT_BKPT();
            var rc = getAndInitPage(pgno, ref pPage);
            if (rc != RC.OK)
                return rc;
            for (var i = 0; i < pPage.Cells; i++)
            {
                var iCell = pPage.FindCell(i);
                var pCell = pPage.Data;
                if (pPage.Leaf == 0)
                {
                    rc = clearDatabasePage(ConvertEx.Get4(pCell, iCell), 1, ref pnChange);
                    if (rc != RC.OK)
                        goto cleardatabasepage_out;
                }
                rc = pPage.clearCell(iCell);
                if (rc != RC.OK)
                    goto cleardatabasepage_out;
            }
            if (pPage.Leaf == 0)
            {
                rc = clearDatabasePage(ConvertEx.Get4(pPage.Data, 8), 1, ref pnChange);
                if (rc != RC.OK)
                    goto cleardatabasepage_out;
            }
            else
                pnChange += pPage.Cells;
            if (freePageFlag != 0)
                pPage.freePage(ref rc);
            else if ((rc = Pager.Write(pPage.DbPage)) == RC.OK)
                pPage.zeroPage(pPage.Data[0] | Btree.PTF_LEAF);
        cleardatabasepage_out:
            pPage.releasePage();
            return rc;
        }

        internal RC btreeGetPage(Pgno pgno, ref MemPage ppPage, int noContent)
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            DbPage pDbPage = null;
            var rc = this.Pager.Get(pgno, ref pDbPage, (byte)noContent);
            if (rc != RC.OK)
                return rc;
            ppPage = MemPage.btreePageFromDbPage(pDbPage, pgno, this);
            return RC.OK;
        }

        internal MemPage btreePageLookup(Pgno pgno)
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            var pDbPage = this.Pager.Lookup(pgno);
            return (pDbPage ? MemPage.btreePageFromDbPage(pDbPage, pgno, this) : null);
        }

        internal Pgno btreePagecount() { return this.Pages; }

        internal RC getAndInitPage(Pgno pgno, ref MemPage ppPage)
        {
            Debug.Assert(MutexEx.Held(this.Mutex));
            RC rc;
            if (pgno > btreePagecount())
                rc = SysEx.SQLITE_CORRUPT_BKPT();
            else
            {
                rc = btreeGetPage(pgno, ref ppPage, 0);
                if (rc == RC.OK)
                {
                    rc = ppPage.btreeInitPage();
                    if (rc != RC.OK)
                        ppPage.releasePage();
                }
            }
            Debug.Assert(pgno != 0 || rc == RC.CORRUPT);
            return rc;
        }
    }
}
