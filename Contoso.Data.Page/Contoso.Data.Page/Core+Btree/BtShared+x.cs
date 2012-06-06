using System;
using System.Diagnostics;
using Contoso.Core.Name;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;
using TRANS = Contoso.Core.Btree.TRANS;

namespace Contoso.Core
{
    public partial class BtShared
    {
        #region Properties

        internal SQLITE btreeSetHasContent(Pgno pgno)
        {
            var rc = SQLITE.OK;
            if (this.pHasContent == null)
            {
                Debug.Assert(pgno <= this.nPage);
                this.pHasContent = new Bitvec(this.nPage);
            }
            if (rc == SQLITE.OK && pgno <= this.pHasContent.sqlite3BitvecSize())
                rc = this.pHasContent.sqlite3BitvecSet(pgno);
            return rc;
        }

        internal bool btreeGetHasContent(Pgno pgno)
        {
            var p = this.pHasContent;
            return (p != null && (pgno > p.sqlite3BitvecSize() || p.sqlite3BitvecTest(pgno) != 0));
        }

        internal void btreeClearHasContent()
        {
            Bitvec.sqlite3BitvecDestroy(ref this.pHasContent);
            this.pHasContent = null;
        }

#if DEBUG
        internal int countWriteCursors()
        {
            var r = 0;
            for (var pCur = this.pCursor; pCur != null; pCur = pCur.pNext)
                if (pCur.wrFlag != 0 && pCur.eState != CURSOR.FAULT)
                    r++;
            return r;
        }
#else
        internal int countWriteCursors() { return -1; }
#endif

        #endregion

        internal SQLITE saveAllCursors(Pgno iRoot, BtCursor pExcept)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            Debug.Assert(pExcept == null || pExcept.pBt == this);
            for (var p = this.pCursor; p != null; p = p.pNext)
                if (p != pExcept && (0 == iRoot || p.pgnoRoot == iRoot) && p.eState == CURSOR.VALID)
                {
                    var rc = p.saveCursorPosition();
                    if (rc != SQLITE.OK)
                        return rc;
                }
            return SQLITE.OK;
        }

#if !SQLITE_OMIT_AUTOVACUUM
        internal Pgno ptrmapPageno(Pgno pgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            if (pgno < 2)
                return 0;
            var nPagesPerMapPage = (int)(this.usableSize / 5 + 1);
            var iPtrMap = (Pgno)((pgno - 2) / nPagesPerMapPage);
            var ret = (Pgno)(iPtrMap * nPagesPerMapPage) + 2;
            if (ret == MemPage.PENDING_BYTE_PAGE(this))
                ret++;
            return ret;
        }

        internal void ptrmapPut(Pgno key, PTRMAP eType, Pgno parent, ref SQLITE pRC)
        {
            if (pRC != 0)
                return;
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            // The master-journal page number must never be used as a pointer map page
            Debug.Assert(!MemPage.PTRMAP_ISPAGE(this, MemPage.PENDING_BYTE_PAGE(this)));
            Debug.Assert(this.autoVacuum);
            if (key == 0)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                return;
            }
            var iPtrmap = MemPage.PTRMAP_PAGENO(this, key);
            var pDbPage = new PgHdr();  // The pointer map page 
            var rc = this.pPager.sqlite3PagerGet(iPtrmap, ref pDbPage);
            if (rc != SQLITE.OK)
            {
                pRC = rc;
                return;
            }
            var offset = (int)MemPage.PTRMAP_PTROFFSET(iPtrmap, key);
            if (offset < 0)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                goto ptrmap_exit;
            }
            Debug.Assert(offset <= (int)this.usableSize - 5);
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage); // The pointer map data 
            if (eType != (PTRMAP)pPtrmap[offset] || ConvertEx.sqlite3Get4byte(pPtrmap, offset + 1) != parent)
            {
                Btree.TRACE("PTRMAP_UPDATE: {0}->({1},{2})", key, eType, parent);
                pRC = rc = Pager.sqlite3PagerWrite(pDbPage);
                if (rc == SQLITE.OK)
                {
                    pPtrmap[offset] = (byte)eType;
                    ConvertEx.sqlite3Put4byte(pPtrmap, offset + 1, parent);
                }
            }
        ptrmap_exit:
            Pager.sqlite3PagerUnref(pDbPage);
        }

        internal SQLITE ptrmapGet(Pgno key, ref PTRMAP pEType, ref Pgno pPgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            var iPtrmap = (int)MemPage.PTRMAP_PAGENO(this, key);
            var pDbPage = new PgHdr(); // The pointer map page
            var rc = this.pPager.sqlite3PagerGet((Pgno)iPtrmap, ref pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage);// Pointer map page data
            var offset = (int)MemPage.PTRMAP_PTROFFSET((Pgno)iPtrmap, key);
            if (offset < 0)
            {
                Pager.sqlite3PagerUnref(pDbPage);
                return SysEx.SQLITE_CORRUPT_BKPT();
            }
            Debug.Assert(offset <= (int)this.usableSize - 5);
            var v = pPtrmap[offset];
            if (v < 1 || v > 5)
                return SysEx.SQLITE_CORRUPT_BKPT();
            pEType = (PTRMAP)v;
            pPgno = ConvertEx.sqlite3Get4byte(pPtrmap, offset + 1);
            Pager.sqlite3PagerUnref(pDbPage);
            return SQLITE.OK;
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
BtShared pList;
bool removed = false;
Debug.Assert( sqlite3_mutex_notheld(pBt.mutex) );
pMaster = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
sqlite3_mutex_enter(pMaster);
pBt.nRef--;
if( pBt.nRef<=0 ){
if( GLOBAL(BtShared*,sqlite3SharedCacheList)==pBt ){
GLOBAL(BtShared*,sqlite3SharedCacheList) = pBt.pNext;
}else{
pList = GLOBAL(BtShared*,sqlite3SharedCacheList);
while( ALWAYS(pList) && pList.pNext!=pBt ){
pList=pList.pNext;
}
if( ALWAYS(pList) ){
pList.pNext = pBt.pNext;
}
}
if( SQLITE_THREADSAFE ){
sqlite3_mutex_free(pBt.mutex);
}
removed = true;
}
sqlite3_mutex_leave(pMaster);
return removed;
#else
            return true;
#endif
        }

        internal void allocateTempSpace() { if (this.pTmpSpace == null) this.pTmpSpace = MallocEx.sqlite3Malloc((int)this.pageSize); }
        internal void freeTempSpace() { PCache1.sqlite3PageFree(ref this.pTmpSpace); }

        internal SQLITE lockBtree()
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            Debug.Assert(this.pPage1 == null);
            var rc = this.pPager.sqlite3PagerSharedLock();
            if (rc != SQLITE.OK)
                return rc;
            MemPage pPage1 = null; // Page 1 of the database file
            rc = btreeGetPage(1, ref pPage1, 0);
            if (rc != SQLITE.OK)
                return rc;
            // Do some checking to help insure the file we opened really is a valid database file.
            Pgno nPageHeader;      // Number of pages in the database according to hdr
            var nPage = nPageHeader = ConvertEx.sqlite3Get4byte(pPage1.aData, 28); // Number of pages in the database
            Pgno nPageFile;    // Number of pages in the database file
            this.pPager.sqlite3PagerPagecount(out nPageFile);
            if (nPage == 0 || ArrayEx.Compare(pPage1.aData, 24, pPage1.aData, 92, 4) != 0)
                nPage = nPageFile;
            if (nPage > 0)
            {
                var page1 = pPage1.aData;
                rc = SQLITE.NOTADB;
                if (ArrayEx.Compare(page1, Btree.zMagicHeader, 16) != 0)
                    goto page1_init_failed;
#if SQLITE_OMIT_WAL
                if (page1[18] > 1)
                    this.readOnly = true;
                if (page1[19] > 1)
                {
                    this.pSchema.file_format = page1[19];
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
                if (pageSize != this.pageSize)
                {
                    // After reading the first page of the database assuming a page size of BtShared.pageSize, we have discovered that the page-size is
                    // actually pageSize. Unlock the database, leave pBt.pPage1 at zero and return SQLITE_OK. The caller will call this function
                    // again with the correct page-size.
                    pPage1.releasePage();
                    this.usableSize = usableSize;
                    this.pageSize = pageSize;
                    rc = this.pPager.sqlite3PagerSetPagesize(ref this.pageSize, (int)(pageSize - usableSize));
                    return rc;
                }
                if ((this.db.flags & sqlite3.SQLITE.RecoveryMode) == 0 && nPage > nPageFile)
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto page1_init_failed;
                }
                if (usableSize < 480)
                    goto page1_init_failed;
                this.pageSize = pageSize;
                this.usableSize = usableSize;
#if !SQLITE_OMIT_AUTOVACUUM
                this.autoVacuum = (ConvertEx.sqlite3Get4byte(page1, 36 + 4 * 4) != 0);
                this.incrVacuum = (ConvertEx.sqlite3Get4byte(page1, 36 + 7 * 4) != 0);
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
            this.maxLocal = (ushort)((this.usableSize - 12) * 64 / 255 - 23);
            this.minLocal = (ushort)((this.usableSize - 12) * 32 / 255 - 23);
            this.maxLeaf = (ushort)(this.usableSize - 35);
            this.minLeaf = (ushort)((this.usableSize - 12) * 32 / 255 - 23);
            Debug.Assert(this.maxLeaf + 23 <= Btree.MX_CELL_SIZE(this));
            this.pPage1 = pPage1;
            this.nPage = nPage;
            return SQLITE.OK;
        page1_init_failed:
            pPage1.releasePage();
            this.pPage1 = null;
            return rc;
        }

        internal void unlockBtreeIfUnused()
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            Debug.Assert(this.pCursor == null || this.inTransaction > TRANS.NONE);
            if (this.inTransaction == TRANS.NONE && this.pPage1 != null)
            {
                Debug.Assert(this.pPage1.aData != null);
                //Debug.Assert(pBt.pPager.sqlite3PagerRefcount() == 1 );
                this.pPage1.releasePage();
                this.pPage1 = null;
            }
        }

        internal SQLITE newDatabase()
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            if (this.nPage > 0)
                return SQLITE.OK;
            var pP1 = this.pPage1;
            Debug.Assert(pP1 != null);
            var data = pP1.aData;
            var rc = Pager.sqlite3PagerWrite(pP1.pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            Buffer.BlockCopy(Btree.zMagicHeader, 0, data, 0, 16);
            Debug.Assert(Btree.zMagicHeader.Length == 16);
            data[16] = (byte)((this.pageSize >> 8) & 0xff);
            data[17] = (byte)((this.pageSize >> 16) & 0xff);
            data[18] = 1;
            data[19] = 1;
            Debug.Assert(this.usableSize <= this.pageSize && this.usableSize + 255 >= this.pageSize);
            data[20] = (byte)(this.pageSize - this.usableSize);
            data[21] = 64;
            data[22] = 32;
            data[23] = 32;
            pP1.zeroPage(Btree.PTF_INTKEY | Btree.PTF_LEAF | Btree.PTF_LEAFDATA);
            this.pageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
            Debug.Assert(this.autoVacuum == true || this.autoVacuum == false);
            Debug.Assert(this.incrVacuum == true || this.incrVacuum == false);
            ConvertEx.sqlite3Put4byte(data, 36 + 4 * 4, this.autoVacuum ? 1 : 0);
            ConvertEx.sqlite3Put4byte(data, 36 + 7 * 4, this.incrVacuum ? 1 : 0);
#endif
            this.nPage = 1;
            data[31] = 1;
            return SQLITE.OK;
        }

        internal SQLITE getOverflowPage(Pgno ovfl, out MemPage ppPage, out Pgno pPgnoNext)
        {
            Pgno next = 0;
            MemPage pPage = null;
            ppPage = null;
            var rc = SQLITE.OK;
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            // Debug.Assert( pPgnoNext != 0);
#if !SQLITE_OMIT_AUTOVACUUM
            // Try to find the next page in the overflow list using the autovacuum pointer-map pages. Guess that the next page in
            // the overflow list is page number (ovfl+1). If that guess turns out to be wrong, fall back to loading the data of page
            // number ovfl to determine the next page number.
            if (this.autoVacuum)
            {
                Pgno pgno = 0;
                Pgno iGuess = ovfl + 1;
                PTRMAP eType = 0;
                while (MemPage.PTRMAP_ISPAGE(this, iGuess) || iGuess == MemPage.PENDING_BYTE_PAGE(this))
                    iGuess++;
                if (iGuess <= btreePagecount())
                {
                    rc = ptrmapGet(iGuess, ref eType, ref pgno);
                    if (rc == SQLITE.OK && eType == PTRMAP.OVERFLOW2 && pgno == ovfl)
                    {
                        next = iGuess;
                        rc = SQLITE.DONE;
                    }
                }
            }
#endif
            Debug.Assert(next == 0 || rc == SQLITE.DONE);
            if (rc == SQLITE.OK)
            {
                rc = btreeGetPage(ovfl, ref pPage, 0);
                Debug.Assert(rc == SQLITE.OK || pPage == null);
                if (rc == SQLITE.OK)
                    next = ConvertEx.sqlite3Get4byte(pPage.aData);
            }
            pPgnoNext = next;
            if (ppPage != null)
                ppPage = pPage;
            else
                pPage.releasePage();
            return (rc == SQLITE.DONE ? SQLITE.OK : rc);
        }

        internal SQLITE clearDatabasePage(Pgno pgno, int freePageFlag, ref int pnChange)
        {
            var pPage = new MemPage();
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            if (pgno > btreePagecount())
                return SysEx.SQLITE_CORRUPT_BKPT();
            var rc = getAndInitPage(pgno, ref pPage);
            if (rc != SQLITE.OK)
                return rc;
            for (var i = 0; i < pPage.nCell; i++)
            {
                var iCell = pPage.findCell(i);
                var pCell = pPage.aData;
                if (pPage.leaf == 0)
                {
                    rc = clearDatabasePage(ConvertEx.sqlite3Get4byte(pCell, iCell), 1, ref pnChange);
                    if (rc != SQLITE.OK)
                        goto cleardatabasepage_out;
                }
                rc = pPage.clearCell(iCell);
                if (rc != SQLITE.OK)
                    goto cleardatabasepage_out;
            }
            if (pPage.leaf == 0)
            {
                rc = clearDatabasePage(ConvertEx.sqlite3Get4byte(pPage.aData, 8), 1, ref pnChange);
                if (rc != SQLITE.OK)
                    goto cleardatabasepage_out;
            }
            else
                pnChange += pPage.nCell;
            if (freePageFlag != 0)
                pPage.freePage(ref rc);
            else if ((rc = Pager.sqlite3PagerWrite(pPage.pDbPage)) == SQLITE.OK)
                pPage.zeroPage(pPage.aData[0] | Btree.PTF_LEAF);
        cleardatabasepage_out:
            pPage.releasePage();
            return rc;
        }

        internal SQLITE btreeGetPage(Pgno pgno, ref MemPage ppPage, int noContent)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            DbPage pDbPage = null;
            var rc = this.pPager.sqlite3PagerAcquire(pgno, ref pDbPage, (byte)noContent);
            if (rc != SQLITE.OK)
                return rc;
            ppPage = MemPage.btreePageFromDbPage(pDbPage, pgno, this);
            return SQLITE.OK;
        }

        internal MemPage btreePageLookup(Pgno pgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            var pDbPage = this.pPager.sqlite3PagerLookup(pgno);
            return (pDbPage ? MemPage.btreePageFromDbPage(pDbPage, pgno, this) : null);
        }

        internal Pgno btreePagecount() { return this.nPage; }

        internal SQLITE getAndInitPage(Pgno pgno, ref MemPage ppPage)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            SQLITE rc;
            if (pgno > btreePagecount())
                rc = SysEx.SQLITE_CORRUPT_BKPT();
            else
            {
                rc = btreeGetPage(pgno, ref ppPage, 0);
                if (rc == SQLITE.OK)
                {
                    rc = ppPage.btreeInitPage();
                    if (rc != SQLITE.OK)
                        ppPage.releasePage();
                }
            }
            Debug.Assert(pgno != 0 || rc == SQLITE.CORRUPT);
            return rc;
        }
    }
}
