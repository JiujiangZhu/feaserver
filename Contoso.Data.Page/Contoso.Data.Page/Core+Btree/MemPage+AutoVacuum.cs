using System.Diagnostics;
using Contoso.Sys;
using Pgno = System.UInt32;
using TRANS = Contoso.Core.Btree.TRANS;

namespace Contoso.Core
{
    public partial class MemPage
    {
#if !SQLITE_OMIT_AUTOVACUUM
        internal SQLITE setChildPtrmaps()
        {
            var pBt = this.pBt;
            var isInitOrig = this.isInit;
            var pgno = this.pgno;
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBt.mutex));
            var rc = btreeInitPage();
            if (rc != SQLITE.OK)
                goto set_child_ptrmaps_out;
            var nCell = this.nCell; // Number of cells in page pPage
            for (var i = 0; i < nCell; i++)
            {
                var pCell = findCell(i);
                ptrmapPutOvflPtr(pCell, ref rc);
                if (this.leaf == 0)
                {
                    var childPgno = (Pgno)ConvertEx.sqlite3Get4byte(this.aData, pCell);
                    pBt.ptrmapPut(childPgno, PTRMAP.BTREE, pgno, ref rc);
                }
            }
            if (this.leaf == 0)
            {
                var childPgno = (Pgno)ConvertEx.sqlite3Get4byte(this.aData, this.hdrOffset + 8);
                pBt.ptrmapPut(childPgno, PTRMAP.BTREE, pgno, ref rc);
            }
        set_child_ptrmaps_out:
            this.isInit = isInitOrig;
            return rc;
        }

        internal SQLITE modifyPagePointer(Pgno iFrom, Pgno iTo, PTRMAP eType)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBt.mutex));
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.pDbPage));
            if (eType == PTRMAP.OVERFLOW2)
            {
                // The pointer is always the first 4 bytes of the page in this case.
                if (ConvertEx.sqlite3Get4byte(this.aData) != iFrom)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                ConvertEx.sqlite3Put4byte(this.aData, iTo);
            }
            else
            {
                var isInitOrig = this.isInit;
                btreeInitPage();
                var nCell = this.nCell;
                int i;
                for (i = 0; i < nCell; i++)
                {
                    var pCell = findCell(i);
                    if (eType == PTRMAP.OVERFLOW1)
                    {
                        var info = new CellInfo();
                        btreeParseCellPtr(pCell, ref info);
                        if (info.iOverflow != 0)
                            if (iFrom == ConvertEx.sqlite3Get4byte(this.aData, pCell, info.iOverflow))
                            {
                                ConvertEx.sqlite3Put4byte(this.aData, pCell + info.iOverflow, (int)iTo);
                                break;
                            }
                    }
                    else
                    {
                        if (ConvertEx.sqlite3Get4byte(this.aData, pCell) == iFrom)
                        {
                            ConvertEx.sqlite3Put4byte(this.aData, pCell, (int)iTo);
                            break;
                        }
                    }
                }
                if (i == nCell)
                {
                    if (eType != PTRMAP.BTREE || ConvertEx.sqlite3Get4byte(this.aData, this.hdrOffset + 8) != iFrom)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    ConvertEx.sqlite3Put4byte(this.aData, this.hdrOffset + 8, iTo);
                }
                this.isInit = isInitOrig;
            }
            return SQLITE.OK;
        }

        internal static SQLITE relocatePage(BtShared pBt, MemPage pDbPage, PTRMAP eType, Pgno iPtrPage, Pgno iFreePage, int isCommit)
        {
            var pPtrPage = new MemPage();   // The page that contains a pointer to pDbPage
            var iDbPage = pDbPage.pgno;
            var pPager = pBt.pPager;
            Debug.Assert(eType == PTRMAP.OVERFLOW2 || eType == PTRMAP.OVERFLOW1 || eType == PTRMAP.BTREE || eType == PTRMAP.ROOTPAGE);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pDbPage.pBt == pBt);
            // Move page iDbPage from its current location to page number iFreePage
            Btree.TRACE("AUTOVACUUM: Moving %d to free page %d (ptr page %d type %d)\n", iDbPage, iFreePage, iPtrPage, eType);
            var rc = pPager.sqlite3PagerMovepage(pDbPage.pDbPage, iFreePage, isCommit);
            if (rc != SQLITE.OK)
                return rc;
            pDbPage.pgno = iFreePage;
            // If pDbPage was a btree-page, then it may have child pages and/or cells that point to overflow pages. The pointer map entries for all these
            // pages need to be changed.
            // If pDbPage is an overflow page, then the first 4 bytes may store a pointer to a subsequent overflow page. If this is the case, then
            // the pointer map needs to be updated for the subsequent overflow page.
            if (eType == PTRMAP.BTREE || eType == PTRMAP.ROOTPAGE)
            {
                rc = pDbPage.setChildPtrmaps();
                if (rc != SQLITE.OK)
                    return rc;
            }
            else
            {
                var nextOvfl = (Pgno)ConvertEx.sqlite3Get4byte(pDbPage.aData);
                if (nextOvfl != 0)
                {
                    pBt.ptrmapPut(nextOvfl, PTRMAP.OVERFLOW2, iFreePage, ref rc);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            }
            // Fix the database pointer on page iPtrPage that pointed at iDbPage so that it points at iFreePage. Also fix the pointer map entry for iPtrPage.
            if (eType != PTRMAP.ROOTPAGE)
            {
                rc = pBt.btreeGetPage(iPtrPage, ref pPtrPage, 0);
                if (rc != SQLITE.OK)
                    return rc;
                rc = Pager.sqlite3PagerWrite(pPtrPage.pDbPage);
                if (rc != SQLITE.OK)
                {
                    pPtrPage.releasePage();
                    return rc;
                }
                rc = pPtrPage.modifyPagePointer(iDbPage, iFreePage, eType);
                pPtrPage.releasePage();
                if (rc == SQLITE.OK)
                    pBt.ptrmapPut(iFreePage, eType, iPtrPage, ref rc);
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
                PTRMAP eType = 0;
                Pgno iPtrPage = 0;
                nFreeList = ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 36);
                if (nFreeList == 0)
                    return SQLITE.DONE;
                var rc = pBt.ptrmapGet( iLastPg, ref eType, ref iPtrPage);
                if (rc != SQLITE.OK)
                    return rc;
                if (eType == PTRMAP.ROOTPAGE)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                if (eType == PTRMAP.FREEPAGE)
                {
                    if (nFin == 0)
                    {
                        // Remove the page from the files free-list. This is not required if nFin is non-zero. In that case, the free-list will be
                        // truncated to zero after this function returns, so it doesn't matter if it still contains some garbage entries.
                        Pgno iFreePg = 0;
                        var pFreePg = new MemPage();
                        rc = pBt.allocateBtreePage( ref pFreePg, ref iFreePg, iLastPg, 1);
                        if (rc != SQLITE.OK)
                            return rc;
                        Debug.Assert(iFreePg == iLastPg);
                        pFreePg.releasePage();
                    }
                }
                else
                {
                    Pgno iFreePg = 0; // Index of free page to move pLastPg to
                    var pLastPg = new MemPage();
                    rc = pBt.btreeGetPage( iLastPg, ref pLastPg, 0);
                    if (rc != SQLITE.OK)
                        return rc;
                    // If nFin is zero, this loop runs exactly once and page pLastPg is swapped with the first free page pulled off the free list.
                    // On the other hand, if nFin is greater than zero, then keep looping until a free-page located within the first nFin pages of the file is found.
                    do
                    {
                        var pFreePg = new MemPage();
                        rc = pBt.allocateBtreePage(ref pFreePg, ref iFreePg, 0, 0);
                        if (rc != SQLITE.OK)
                        {
                            pLastPg.releasePage();
                            return rc;
                        }
                        pFreePg.releasePage();
                    } while (nFin != 0 && iFreePg > nFin);
                    Debug.Assert(iFreePg < iLastPg);
                    rc = Pager.sqlite3PagerWrite(pLastPg.pDbPage);
                    if (rc == SQLITE.OK)
                        rc = relocatePage(pBt, pLastPg, eType, iPtrPage, iFreePg, (nFin != 0) ? 1 : 0);
                    pLastPg.releasePage();
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
                        var rc = pBt.btreeGetPage(iLastPg, ref pPg, 0);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = Pager.sqlite3PagerWrite(pPg.pDbPage);
                        pPg.releasePage();
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
            p.sqlite3BtreeEnter();
            Debug.Assert(pBt.inTransaction == TRANS.WRITE && p.inTrans == TRANS.WRITE);
            SQLITE rc;
            if (!pBt.autoVacuum)
                rc = SQLITE.DONE;
            else
            {
                Btree.invalidateAllOverflowCache(pBt);
                rc = incrVacuumStep(pBt, 0, pBt.btreePagecount());
                if (rc == SQLITE.OK)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.pPage1.pDbPage);
                    ConvertEx.sqlite3Put4byte(pBt.pPage1.aData, (uint)28, pBt.nPage);
                }
            }
            p.sqlite3BtreeLeave();
            return rc;
        }

        internal static SQLITE autoVacuumCommit(BtShared pBt)
        {
            var rc = SQLITE.OK;
            var pPager = pBt.pPager;
#if DEBUG
            var nRef = pPager.sqlite3PagerRefcount();
#else
            var nRef = 0;
#endif
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Btree.invalidateAllOverflowCache(pBt);
            Debug.Assert(pBt.autoVacuum);
            if (!pBt.incrVacuum)
            {
                var nOrig = pBt.btreePagecount(); // Database size before freeing
                if (PTRMAP_ISPAGE(pBt, nOrig) || nOrig == PENDING_BYTE_PAGE(pBt))
                    // It is not possible to create a database for which the final page is either a pointer-map page or the pending-byte page. If one
                    // is encountered, this indicates corruption.
                    return SysEx.SQLITE_CORRUPT_BKPT();
                var nFree = (Pgno)ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 36); // Number of pages on the freelist initially
                var nEntry = (int)pBt.usableSize / 5; // Number of entries on one ptrmap page
                var nPtrmap = (Pgno)((nFree - nOrig + PTRMAP_PAGENO(pBt, nOrig) + (Pgno)nEntry) / nEntry); // Number of PtrMap pages to be freed 
                var nFin = nOrig - nFree - nPtrmap; // Number of pages in database after autovacuuming
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
        internal SQLITE setChildPtrmaps() { return SQLITE.OK; }
#endif
    }
}
