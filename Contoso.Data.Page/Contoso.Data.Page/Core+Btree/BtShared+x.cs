using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;

namespace Contoso.Core
{
    public partial class BtShared
    {
        #region Properties

        internal static SQLITE btreeSetHasContent(BtShared pBt, Pgno pgno)
        {
            var rc = SQLITE.OK;
            if (pBt.pHasContent == null)
            {
                Debug.Assert(pgno <= pBt.nPage);
                pBt.pHasContent = new Bitvec(pBt.nPage);
            }
            if (rc == SQLITE.OK && pgno <= pBt.pHasContent.sqlite3BitvecSize())
                rc = pBt.pHasContent.sqlite3BitvecSet(pgno);
            return rc;
        }

        internal static bool btreeGetHasContent(BtShared pBt, Pgno pgno)
        {
            var p = pBt.pHasContent;
            return (p != null && (pgno > p.sqlite3BitvecSize() || p.sqlite3BitvecTest(pgno) != 0));
        }

        internal static void btreeClearHasContent(BtShared pBt)
        {
            Bitvec.sqlite3BitvecDestroy(ref pBt.pHasContent);
            pBt.pHasContent = null;
        }

        #endregion


        internal static SQLITE saveAllCursors(BtShared pBt, Pgno iRoot, BtCursor pExcept)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pExcept == null || pExcept.pBt == pBt);
            for (var p = pBt.pCursor; p != null; p = p.pNext)
                if (p != pExcept && (0 == iRoot || p.pgnoRoot == iRoot) && p.eState == CURSOR.VALID)
                {
                    var rc = saveCursorPosition(p);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            return SQLITE.OK;
        }

#if !SQLITE_OMIT_AUTOVACUUM
        internal static Pgno ptrmapPageno(BtShared pBt, Pgno pgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            if (pgno < 2)
                return 0;
            var nPagesPerMapPage = (int)(pBt.usableSize / 5 + 1);
            var iPtrMap = (Pgno)((pgno - 2) / nPagesPerMapPage);
            var ret = (Pgno)(iPtrMap * nPagesPerMapPage) + 2;
            if (ret == PENDING_BYTE_PAGE(pBt))
                ret++;
            return ret;
        }

        internal static void ptrmapPut(BtShared pBt, Pgno key, byte eType, Pgno parent, ref SQLITE pRC)
        {
            if (pRC != 0)
                return;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            // The master-journal page number must never be used as a pointer map page
            Debug.Assert(!PTRMAP_ISPAGE(pBt, PENDING_BYTE_PAGE(pBt)));
            Debug.Assert(pBt.autoVacuum);
            if (key == 0)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                return;
            }
            var iPtrmap = PTRMAP_PAGENO(pBt, key);
            var pDbPage = new PgHdr();  // The pointer map page 
            var rc = pBt.pPager.sqlite3PagerGet(iPtrmap, ref pDbPage);
            if (rc != SQLITE.OK)
            {
                pRC = rc;
                return;
            }
            var offset = (int)PTRMAP_PTROFFSET(iPtrmap, key);
            if (offset < 0)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                goto ptrmap_exit;
            }
            Debug.Assert(offset <= (int)pBt.usableSize - 5);
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage); // The pointer map data 
            if (eType != pPtrmap[offset] || ConvertEx.sqlite3Get4byte(pPtrmap, offset + 1) != parent)
            {
                TRACE("PTRMAP_UPDATE: {0}->({1},{2})", key, eType, parent);
                pRC = rc = Pager.sqlite3PagerWrite(pDbPage);
                if (rc == SQLITE.OK)
                {
                    pPtrmap[offset] = eType;
                    ConvertEx.sqlite3Put4byte(pPtrmap, offset + 1, parent);
                }
            }
        ptrmap_exit:
            Pager.sqlite3PagerUnref(pDbPage);
        }

        internal static SQLITE ptrmapGet(BtShared pBt, Pgno key, ref byte pEType, ref Pgno pPgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            var iPtrmap = (int)PTRMAP_PAGENO(pBt, key);
            var pDbPage = new PgHdr(); // The pointer map page
            var rc = pBt.pPager.sqlite3PagerGet((Pgno)iPtrmap, ref pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage);// Pointer map page data
            var offset = (int)PTRMAP_PTROFFSET((Pgno)iPtrmap, key);
            if (offset < 0)
            {
                Pager.sqlite3PagerUnref(pDbPage);
                return SysEx.SQLITE_CORRUPT_BKPT();
            }
            Debug.Assert(offset <= (int)pBt.usableSize - 5);
            pEType = pPtrmap[offset];
            pPgno = ConvertEx.sqlite3Get4byte(pPtrmap, offset + 1);
            Pager.sqlite3PagerUnref(pDbPage);
            if (pEType < 1 || pEType > 5)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return SQLITE.OK;
        }

#else
        internal static Pgno ptrmapPageno(BtShared pBt, Pgno pgno) { return 0; }
        internal static void ptrmapPut(BtShared pBt, Pgno key, byte eType, Pgno parent, ref SQLITE pRC) { pRC = SQLITE.OK; }
        internal static SQLITE ptrmapGet(BtShared pBt, Pgno key, ref byte pEType, ref Pgno pPgno) { return SQLITE.OK; }
#endif

        internal static bool removeFromSharingList(BtShared pBt)
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

        internal static void allocateTempSpace(BtShared pBt) { if (pBt.pTmpSpace == null) pBt.pTmpSpace = MallocEx.sqlite3Malloc((int)pBt.pageSize); }
        internal static void freeTempSpace(BtShared pBt) { Pager.sqlite3PageFree(ref pBt.pTmpSpace); }

        internal static SQLITE lockBtree(BtShared pBt)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pBt.pPage1 == null);
            var rc = pBt.pPager.sqlite3PagerSharedLock();
            if (rc != SQLITE.OK)
                return rc;
            MemPage pPage1 = null; // Page 1 of the database file
            rc = btreeGetPage(pBt, 1, ref pPage1, 0);
            if (rc != SQLITE.OK)
                return rc;
            // Do some checking to help insure the file we opened really is a valid database file.
            Pgno nPageHeader;      // Number of pages in the database according to hdr
            var nPage = nPageHeader = ConvertEx.sqlite3Get4byte(pPage1.aData, 28); // Number of pages in the database
            Pgno nPageFile;    // Number of pages in the database file
            pBt.pPager.sqlite3PagerPagecount(out nPageFile);
            if (nPage == 0 || ArrayEx.Compare(pPage1.aData, 24, pPage1.aData, 92, 4) != 0)
                nPage = nPageFile;
            if (nPage > 0)
            {
                var page1 = pPage1.aData;
                rc = SQLITE.NOTADB;
                if (ArrayEx.Compare(page1, zMagicHeader, 16) != 0)
                    goto page1_init_failed;
#if SQLITE_OMIT_WAL
                if (page1[18] > 1)
                    pBt.readOnly = true;
                if (page1[19] > 1)
                {
                    pBt.pSchema.file_format = page1[19];
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
                if (ArrayEx.Compare(page1, 21, "\x0040\x0020\x0020", 3) != 0)//   "\100\040\040"
                    goto page1_init_failed;
                var pageSize = (uint)((page1[16] << 8) | (page1[17] << 16));
                if (((pageSize - 1) & pageSize) != 0 || pageSize > Pager.SQLITE_MAX_PAGE_SIZE || pageSize <= 256)
                    goto page1_init_failed;
                Debug.Assert((pageSize & 7) == 0);
                var usableSize = pageSize - page1[20];
                if (pageSize != pBt.pageSize)
                {
                    // After reading the first page of the database assuming a page size of BtShared.pageSize, we have discovered that the page-size is
                    // actually pageSize. Unlock the database, leave pBt.pPage1 at zero and return SQLITE_OK. The caller will call this function
                    // again with the correct page-size.
                    releasePage(pPage1);
                    pBt.usableSize = usableSize;
                    pBt.pageSize = pageSize;
                    rc = pBt.pPager.sqlite3PagerSetPagesize(ref pBt.pageSize, (int)(pageSize - usableSize));
                    return rc;
                }
                if ((pBt.db.flags & SQLITE_RecoveryMode) == 0 && nPage > nPageFile)
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto page1_init_failed;
                }
                if (usableSize < 480)
                    goto page1_init_failed;
                pBt.pageSize = pageSize;
                pBt.usableSize = usableSize;
#if !SQLITE_OMIT_AUTOVACUUM
                pBt.autoVacuum = (ConvertEx.sqlite3Get4byte(page1, 36 + 4 * 4) != 0);
                pBt.incrVacuum = (ConvertEx.sqlite3Get4byte(page1, 36 + 7 * 4) != 0);
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
            pBt.maxLocal = (ushort)((pBt.usableSize - 12) * 64 / 255 - 23);
            pBt.minLocal = (ushort)((pBt.usableSize - 12) * 32 / 255 - 23);
            pBt.maxLeaf = (ushort)(pBt.usableSize - 35);
            pBt.minLeaf = (ushort)((pBt.usableSize - 12) * 32 / 255 - 23);
            Debug.Assert(pBt.maxLeaf + 23 <= MX_CELL_SIZE(pBt));
            pBt.pPage1 = pPage1;
            pBt.nPage = nPage;
            return SQLITE.OK;
        page1_init_failed:
            releasePage(pPage1);
            pBt.pPage1 = null;
            return rc;
        }

        internal static void unlockBtreeIfUnused(BtShared pBt)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pBt.pCursor == null || pBt.inTransaction > TRANS.NONE);
            if (pBt.inTransaction == TRANS.NONE && pBt.pPage1 != null)
            {
                Debug.Assert(pBt.pPage1.aData != null);
                //Debug.Assert(pBt.pPager.sqlite3PagerRefcount() == 1 );
                releasePage(pBt.pPage1);
                pBt.pPage1 = null;
            }
        }

        internal static SQLITE newDatabase(BtShared pBt)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            if (pBt.nPage > 0)
                return SQLITE.OK;
            var pP1 = pBt.pPage1;
            Debug.Assert(pP1 != null);
            var data = pP1.aData;
            var rc = Pager.sqlite3PagerWrite(pP1.pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            Buffer.BlockCopy(zMagicHeader, 0, data, 0, 16);
            Debug.Assert(zMagicHeader.Length == 16);
            data[16] = (byte)((pBt.pageSize >> 8) & 0xff);
            data[17] = (byte)((pBt.pageSize >> 16) & 0xff);
            data[18] = 1;
            data[19] = 1;
            Debug.Assert(pBt.usableSize <= pBt.pageSize && pBt.usableSize + 255 >= pBt.pageSize);
            data[20] = (byte)(pBt.pageSize - pBt.usableSize);
            data[21] = 64;
            data[22] = 32;
            data[23] = 32;
            zeroPage(pP1, PTF_INTKEY | PTF_LEAF | PTF_LEAFDATA);
            pBt.pageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
            Debug.Assert(pBt.autoVacuum == true || pBt.autoVacuum == false);
            Debug.Assert(pBt.incrVacuum == true || pBt.incrVacuum == false);
            ConvertEx.sqlite3Put4byte(data, 36 + 4 * 4, pBt.autoVacuum ? 1 : 0);
            ConvertEx.sqlite3Put4byte(data, 36 + 7 * 4, pBt.incrVacuum ? 1 : 0);
#endif
            pBt.nPage = 1;
            data[31] = 1;
            return SQLITE.OK;
        }

        internal static SQLITE getOverflowPage(BtShared pBt, Pgno ovfl, out MemPage ppPage, out Pgno pPgnoNext)
        {
            Pgno next = 0;
            MemPage pPage = null;
            ppPage = null;
            var rc = SQLITE.OK;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            // Debug.Assert( pPgnoNext != 0);
#if !SQLITE_OMIT_AUTOVACUUM
            // Try to find the next page in the overflow list using the autovacuum pointer-map pages. Guess that the next page in
            // the overflow list is page number (ovfl+1). If that guess turns out to be wrong, fall back to loading the data of page
            // number ovfl to determine the next page number.
            if (pBt.autoVacuum)
            {
                Pgno pgno = 0;
                Pgno iGuess = ovfl + 1;
                byte eType = 0;
                while (PTRMAP_ISPAGE(pBt, iGuess) || iGuess == PENDING_BYTE_PAGE(pBt))
                    iGuess++;
                if (iGuess <= btreePagecount(pBt))
                {
                    rc = ptrmapGet(pBt, iGuess, ref eType, ref pgno);
                    if (rc == SQLITE.OK && eType == PTRMAP_OVERFLOW2 && pgno == ovfl)
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
                rc = btreeGetPage(pBt, ovfl, ref pPage, 0);
                Debug.Assert(rc == SQLITE.OK || pPage == null);
                if (rc == SQLITE.OK)
                    next = ConvertEx.sqlite3Get4byte(pPage.aData);
            }
            pPgnoNext = next;
            if (ppPage != null)
                ppPage = pPage;
            else
                releasePage(pPage);
            return (rc == SQLITE.DONE ? SQLITE.OK : rc);
        }
        
        internal static SQLITE clearDatabasePage(BtShared pBt,Pgno pgno,int freePageFlag,     ref int pnChange)
        {
            var pPage = new MemPage();
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            if (pgno > btreePagecount(pBt))
                return SysEx.SQLITE_CORRUPT_BKPT();
            var rc = getAndInitPage(pBt, pgno, ref pPage);
            if (rc != SQLITE.OK)
                return rc;
            for (var i = 0; i < pPage.nCell; i++)
            {
                var iCell = findCell(pPage, i);
                var pCell = pPage.aData;
                if (pPage.leaf == 0)
                {
                    rc = clearDatabasePage(pBt, ConvertEx.sqlite3Get4byte(pCell, iCell), 1, ref pnChange);
                    if (rc != SQLITE.OK)
                        goto cleardatabasepage_out;
                }
                rc = clearCell(pPage, iCell);
                if (rc != SQLITE.OK)
                    goto cleardatabasepage_out;
            }
            if (pPage.leaf == 0)
            {
                rc = clearDatabasePage(pBt, ConvertEx.sqlite3Get4byte(pPage.aData, 8), 1, ref pnChange);
                if (rc != SQLITE.OK)
                    goto cleardatabasepage_out;
            }
            else
                pnChange += pPage.nCell;
            if (freePageFlag != 0)
                freePage(pPage, ref rc);
            else if ((rc = Pager.sqlite3PagerWrite(pPage.pDbPage)) == SQLITE.OK)
                zeroPage(pPage, pPage.aData[0] | PTF_LEAF);
        cleardatabasepage_out:
            releasePage(pPage);
            return rc;
        }

    }
}
