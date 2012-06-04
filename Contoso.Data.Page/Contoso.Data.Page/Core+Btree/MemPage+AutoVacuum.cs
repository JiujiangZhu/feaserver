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
    public partial class MemPage
    {
#if !SQLITE_OMIT_AUTOVACUUM
        internal static SQLITE setChildPtrmaps(MemPage pPage)
        {
            var pBt = pPage.pBt;
            var isInitOrig = pPage.isInit;
            var pgno = pPage.pgno;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            var rc = btreeInitPage(pPage);
            if (rc != SQLITE.OK)
                goto set_child_ptrmaps_out;
            var nCell = pPage.nCell; // Number of cells in page pPage
            for (var i = 0; i < nCell; i++)
            {
                int pCell = findCell(pPage, i);
                ptrmapPutOvflPtr(pPage, pCell, ref rc);
                if (pPage.leaf == 0)
                {
                    var childPgno = (Pgno)ConvertEx.sqlite3Get4byte(pPage.aData, pCell);
                    ptrmapPut(pBt, childPgno, PTRMAP_BTREE, pgno, ref rc);
                }
            }
            if (pPage.leaf == 0)
            {
                Pgno childPgno = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                ptrmapPut(pBt, childPgno, PTRMAP_BTREE, pgno, ref rc);
            }
        set_child_ptrmaps_out:
            pPage.isInit = isInitOrig;
            return rc;
        }

        internal static SQLITE modifyPagePointer(MemPage pPage, Pgno iFrom, Pgno iTo, byte eType)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            if (eType == PTRMAP_OVERFLOW2)
            {
                // The pointer is always the first 4 bytes of the page in this case.
                if (ConvertEx.sqlite3Get4byte(pPage.aData) != iFrom)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                ConvertEx.sqlite3Put4byte(pPage.aData, iTo);
            }
            else
            {
                var isInitOrig = pPage.isInit;
                btreeInitPage(pPage);
                var nCell = pPage.nCell;
                for (var i = 0; i < nCell; i++)
                {
                    var pCell = findCell(pPage, i);
                    if (eType == PTRMAP_OVERFLOW1)
                    {
                        var info = new CellInfo();
                        btreeParseCellPtr(pPage, pCell, ref info);
                        if (info.iOverflow != 0)
                            if (iFrom == ConvertEx.sqlite3Get4byte(pPage.aData, pCell, info.iOverflow))
                            {
                                ConvertEx.sqlite3Put4byte(pPage.aData, pCell + info.iOverflow, (int)iTo);
                                break;
                            }
                    }
                    else
                    {
                        if (ConvertEx.sqlite3Get4byte(pPage.aData, pCell) == iFrom)
                        {
                            ConvertEx.sqlite3Put4byte(pPage.aData, pCell, (int)iTo);
                            break;
                        }
                    }
                }
                if (i == nCell)
                {
                    if (eType != PTRMAP_BTREE || ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8) != iFrom)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    ConvertEx.sqlite3Put4byte(pPage.aData, pPage.hdrOffset + 8, iTo);
                }
                pPage.isInit = isInitOrig;
            }
            return SQLITE.OK;
        }

        internal static SQLITE relocatePage(BtShared pBt, MemPage pDbPage, byte eType, Pgno iPtrPage, Pgno iFreePage, int isCommit)
        {
            var pPtrPage = new MemPage();   // The page that contains a pointer to pDbPage
            var iDbPage = pDbPage.pgno;
            var pPager = pBt.pPager;
            Debug.Assert(eType == PTRMAP_OVERFLOW2 || eType == PTRMAP_OVERFLOW1 || eType == PTRMAP_BTREE || eType == PTRMAP_ROOTPAGE);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pDbPage.pBt == pBt);
            // Move page iDbPage from its current location to page number iFreePage
            TRACE("AUTOVACUUM: Moving %d to free page %d (ptr page %d type %d)\n", iDbPage, iFreePage, iPtrPage, eType);
            var rc = pPager.sqlite3PagerMovepage(pDbPage.pDbPage, iFreePage, isCommit);
            if (rc != SQLITE.OK)
                return rc;
            pDbPage.pgno = iFreePage;
            // If pDbPage was a btree-page, then it may have child pages and/or cells that point to overflow pages. The pointer map entries for all these
            // pages need to be changed.
            // If pDbPage is an overflow page, then the first 4 bytes may store a pointer to a subsequent overflow page. If this is the case, then
            // the pointer map needs to be updated for the subsequent overflow page.
            if (eType == PTRMAP_BTREE || eType == PTRMAP_ROOTPAGE)
            {
                rc = setChildPtrmaps(pDbPage);
                if (rc != SQLITE.OK)
                    return rc;
            }
            else
            {
                var nextOvfl = (Pgno)ConvertEx.sqlite3Get4byte(pDbPage.aData);
                if (nextOvfl != 0)
                {
                    ptrmapPut(pBt, nextOvfl, PTRMAP_OVERFLOW2, iFreePage, ref rc);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            }
            // Fix the database pointer on page iPtrPage that pointed at iDbPage so that it points at iFreePage. Also fix the pointer map entry for iPtrPage.
            if (eType != PTRMAP_ROOTPAGE)
            {
                rc = btreeGetPage(pBt, iPtrPage, ref pPtrPage, 0);
                if (rc != SQLITE.OK)
                    return rc;
                rc = Pager.sqlite3PagerWrite(pPtrPage.pDbPage);
                if (rc != SQLITE.OK)
                {
                    releasePage(pPtrPage);
                    return rc;
                }
                rc = modifyPagePointer(pPtrPage, iDbPage, iFreePage, eType);
                releasePage(pPtrPage);
                if (rc == SQLITE.OK)
                    ptrmapPut(pBt, iFreePage, eType, iPtrPage, ref rc);
            }
            return rc;
        }

        internal static SQLITE incrVacuumStep(BtShared pBt, Pgno nFin, Pgno iLastPg)
        {
            Pgno nFreeList;           // Number of pages still on the free-list
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(iLastPg > nFin);
            if (!PTRMAP_ISPAGE(pBt, iLastPg) && iLastPg != PENDING_BYTE_PAGE(pBt))
            {
                byte eType = 0;
                Pgno iPtrPage = 0;
                nFreeList = ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 36);
                if (nFreeList == 0)
                    return SQLITE.DONE;
                var rc = ptrmapGet(pBt, iLastPg, ref eType, ref iPtrPage);
                if (rc != SQLITE.OK)
                    return rc;
                if (eType == PTRMAP_ROOTPAGE)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                if (eType == PTRMAP_FREEPAGE)
                {
                    if (nFin == 0)
                    {
                        // Remove the page from the files free-list. This is not required if nFin is non-zero. In that case, the free-list will be
                        // truncated to zero after this function returns, so it doesn't matter if it still contains some garbage entries.
                        Pgno iFreePg = 0;
                        var pFreePg = new MemPage();
                        rc = allocateBtreePage(pBt, ref pFreePg, ref iFreePg, iLastPg, 1);
                        if (rc != SQLITE.OK)
                            return rc;
                        Debug.Assert(iFreePg == iLastPg);
                        releasePage(pFreePg);
                    }
                }
                else
                {
                    Pgno iFreePg = 0; // Index of free page to move pLastPg to
                    var pLastPg = new MemPage();
                    rc = btreeGetPage(pBt, iLastPg, ref pLastPg, 0);
                    if (rc != SQLITE.OK)
                        return rc;
                    // If nFin is zero, this loop runs exactly once and page pLastPg is swapped with the first free page pulled off the free list.
                    // On the other hand, if nFin is greater than zero, then keep looping until a free-page located within the first nFin pages of the file is found.
                    do
                    {
                        var pFreePg = new MemPage();
                        rc = allocateBtreePage(pBt, ref pFreePg, ref iFreePg, 0, 0);
                        if (rc != SQLITE.OK)
                        {
                            releasePage(pLastPg);
                            return rc;
                        }
                        releasePage(pFreePg);
                    } while (nFin != 0 && iFreePg > nFin);
                    Debug.Assert(iFreePg < iLastPg);
                    rc = Pager.sqlite3PagerWrite(pLastPg.pDbPage);
                    if (rc == SQLITE.OK)
                        rc = relocatePage(pBt, pLastPg, eType, iPtrPage, iFreePg, (nFin != 0) ? 1 : 0);
                    releasePage(pLastPg);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            }
            if (nFin == 0)
            {
                iLastPg--;
                while (iLastPg == PENDING_BYTE_PAGE(pBt) || PTRMAP_ISPAGE(pBt, iLastPg))
                {
                    if (PTRMAP_ISPAGE(pBt, iLastPg))
                    {
                        var pPg = new MemPage();
                        var rc = btreeGetPage(pBt, iLastPg, ref pPg, 0);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = Pager.sqlite3PagerWrite(pPg.pDbPage);
                        releasePage(pPg);
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                    iLastPg--;
                }
                pBt.pPager.sqlite3PagerTruncateImage(iLastPg);
                pBt.nPage = iLastPg;
            }
            return SQLITE.OK;
        }

        internal static SQLITE sqlite3BtreeIncrVacuum(Btree p)
        {
            var pBt = p.pBt;
            sqlite3BtreeEnter(p);
            Debug.Assert(pBt.inTransaction == TRANS.WRITE && p.inTrans == TRANS.WRITE);
            SQLITE rc;
            if (!pBt.autoVacuum)
                rc = SQLITE.DONE;
            else
            {
                invalidateAllOverflowCache(pBt);
                rc = incrVacuumStep(pBt, 0, btreePagecount(pBt));
                if (rc == SQLITE.OK)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.pPage1.pDbPage);
                    ConvertEx.sqlite3Put4byte(pBt.pPage1.aData, (uint)28, pBt.nPage);
                }
            }
            sqlite3BtreeLeave(p);
            return rc;
        }

        internal static SQLITE autoVacuumCommit(BtShared pBt)
        {
            var rc = SQLITE.OK;
            var pPager = pBt.pPager;
#if DEBUG
            int nRef = pPager.sqlite3PagerRefcount();
#else
            int nRef = 0;
#endif
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            invalidateAllOverflowCache(pBt);
            Debug.Assert(pBt.autoVacuum);
            if (!pBt.incrVacuum)
            {
                var nOrig = btreePagecount(pBt); // Database size before freeing
                if (PTRMAP_ISPAGE(pBt, nOrig) || nOrig == PENDING_BYTE_PAGE(pBt))
                    // It is not possible to create a database for which the final page is either a pointer-map page or the pending-byte page. If one
                    // is encountered, this indicates corruption.
                    return SysEx.SQLITE_CORRUPT_BKPT();
                var nFree = (Pgno)ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 36); // Number of pages on the freelist initially
                var nEntry = (int)pBt.usableSize / 5; // Number of entries on one ptrmap page
                nPtrmap = (Pgno)((nFree - nOrig + PTRMAP_PAGENO(pBt, nOrig) + (Pgno)nEntry) / nEntry); // Number of PtrMap pages to be freed 
                nFin = nOrig - nFree - nPtrmap; // Number of pages in database after autovacuuming
                if (nOrig > PENDING_BYTE_PAGE(pBt) && nFin < PENDING_BYTE_PAGE(pBt))
                    nFin--;
                while (PTRMAP_ISPAGE(pBt, nFin) || nFin == PENDING_BYTE_PAGE(pBt))
                    nFin--;
                if (nFin > nOrig)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                for (var iFree = nOrig; iFree > nFin && rc == SQLITE.OK; iFree--)
                    rc = incrVacuumStep(pBt, nFin, iFree);
                if ((rc == SQLITE.DONE || rc == SQLITE.OK) && nFree > 0)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.pPage1.pDbPage);
                    ConvertEx.sqlite3Put4byte(pBt.pPage1.aData, 32, 0);
                    ConvertEx.sqlite3Put4byte(pBt.pPage1.aData, 36, 0);
                    ConvertEx.sqlite3Put4byte(pBt.pPage1.aData, (uint)28, nFin);
                    pBt.pPager.sqlite3PagerTruncateImage(nFin);
                    pBt.nPage = nFin;
                }
                if (rc != SQLITE.OK)
                    pPager.sqlite3PagerRollback();
            }
            Debug.Assert(nRef == pPager.sqlite3PagerRefcount());
            return rc;
        }
#else
        internal static int setChildPtrmaps(MemPage pPage) { return SQLITE.OK; }
#endif
    }
}
