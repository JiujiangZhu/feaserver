using System.Diagnostics;
using Contoso.Sys;
using Pgno = System.UInt32;
using TRANS = Contoso.Core.Btree.TRANS;

namespace Contoso.Core
{
    public partial class MemPage
    {
#if !SQLITE_OMIT_AUTOVACUUM
        internal void ptrmapPutOvflPtr(int pCell, ref RC pRC)
        {
            if (pRC != 0)
                return;
            var info = new CellInfo();
            Debug.Assert(pCell != 0);
            btreeParseCellPtr(pCell, ref info);
            Debug.Assert((info.nData + (this.HasIntKey ? 0 : info.nKey)) == info.nPayload);
            if (info.iOverflow != 0)
            {
                Pgno ovfl = ConvertEx.Get4(this.Data, pCell, info.iOverflow);
                this.Shared.ptrmapPut(ovfl, PTRMAP.OVERFLOW1, this.ID, ref pRC);
            }
        }

        internal void ptrmapPutOvflPtr(byte[] pCell, ref RC pRC)
        {
            if (pRC != 0)
                return;
            var info = new CellInfo();
            Debug.Assert(pCell != null);
            btreeParseCellPtr(pCell, ref info);
            Debug.Assert((info.nData + (this.HasIntKey ? 0 : info.nKey)) == info.nPayload);
            if (info.iOverflow != 0)
            {
                Pgno ovfl = ConvertEx.Get4(pCell, info.iOverflow);
                this.Shared.ptrmapPut(ovfl, PTRMAP.OVERFLOW1, this.ID, ref pRC);
            }
        }
#endif

#if !SQLITE_OMIT_AUTOVACUUM
        internal RC setChildPtrmaps()
        {
            var hasInitLast = HasInit;
            Debug.Assert(MutexEx.Held(Shared.Mutex));
            var rc = btreeInitPage();
            if (rc != RC.OK)
                goto set_child_ptrmaps_out;
            var cells = Cells; // Number of cells in page pPage
            var shared = Shared;
            var id = ID;
            for (var i = 0; i < cells; i++)
            {
                var cell = FindCell(i);
                ptrmapPutOvflPtr(cell, ref rc);
                if (Leaf == 0)
                {
                    var childPgno = (Pgno)ConvertEx.Get4(Data, cell);
                    shared.ptrmapPut(childPgno, PTRMAP.BTREE, id, ref rc);
                }
            }
            if (Leaf == 0)
            {
                var childPgno = (Pgno)ConvertEx.Get4(Data, HeaderOffset + 8);
                shared.ptrmapPut(childPgno, PTRMAP.BTREE, id, ref rc);
            }
        set_child_ptrmaps_out:
            HasInit = hasInitLast;
            return rc;
        }

        internal RC modifyPagePointer(Pgno iFrom, Pgno iTo, PTRMAP eType)
        {
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            if (eType == PTRMAP.OVERFLOW2)
            {
                // The pointer is always the first 4 bytes of the page in this case.
                if (ConvertEx.Get4(this.Data) != iFrom)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                ConvertEx.Put4L(this.Data, iTo);
            }
            else
            {
                var isInitOrig = this.HasInit;
                btreeInitPage();
                var nCell = this.Cells;
                int i;
                for (i = 0; i < nCell; i++)
                {
                    var pCell = FindCell(i);
                    if (eType == PTRMAP.OVERFLOW1)
                    {
                        var info = new CellInfo();
                        btreeParseCellPtr(pCell, ref info);
                        if (info.iOverflow != 0)
                            if (iFrom == ConvertEx.Get4(this.Data, pCell, info.iOverflow))
                            {
                                ConvertEx.Put4(this.Data, pCell + info.iOverflow, (int)iTo);
                                break;
                            }
                    }
                    else
                    {
                        if (ConvertEx.Get4(this.Data, pCell) == iFrom)
                        {
                            ConvertEx.Put4(this.Data, pCell, (int)iTo);
                            break;
                        }
                    }
                }
                if (i == nCell)
                {
                    if (eType != PTRMAP.BTREE || ConvertEx.Get4(this.Data, this.HeaderOffset + 8) != iFrom)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    ConvertEx.Put4L(this.Data, this.HeaderOffset + 8, iTo);
                }
                this.HasInit = isInitOrig;
            }
            return RC.OK;
        }

        internal static RC relocatePage(BtShared pBt, MemPage pDbPage, PTRMAP eType, Pgno iPtrPage, Pgno iFreePage, int isCommit)
        {
            var pPtrPage = new MemPage();   // The page that contains a pointer to pDbPage
            var iDbPage = pDbPage.ID;
            var pPager = pBt.Pager;
            Debug.Assert(eType == PTRMAP.OVERFLOW2 || eType == PTRMAP.OVERFLOW1 || eType == PTRMAP.BTREE || eType == PTRMAP.ROOTPAGE);
            Debug.Assert(MutexEx.Held(pBt.Mutex));
            Debug.Assert(pDbPage.Shared == pBt);
            // Move page iDbPage from its current location to page number iFreePage
            Btree.TRACE("AUTOVACUUM: Moving %d to free page %d (ptr page %d type %d)\n", iDbPage, iFreePage, iPtrPage, eType);
            var rc = pPager.sqlite3PagerMovepage(pDbPage.DbPage, iFreePage, isCommit);
            if (rc != RC.OK)
                return rc;
            pDbPage.ID = iFreePage;
            // If pDbPage was a btree-page, then it may have child pages and/or cells that point to overflow pages. The pointer map entries for all these
            // pages need to be changed.
            // If pDbPage is an overflow page, then the first 4 bytes may store a pointer to a subsequent overflow page. If this is the case, then
            // the pointer map needs to be updated for the subsequent overflow page.
            if (eType == PTRMAP.BTREE || eType == PTRMAP.ROOTPAGE)
            {
                rc = pDbPage.setChildPtrmaps();
                if (rc != RC.OK)
                    return rc;
            }
            else
            {
                var nextOvfl = (Pgno)ConvertEx.Get4(pDbPage.Data);
                if (nextOvfl != 0)
                {
                    pBt.ptrmapPut(nextOvfl, PTRMAP.OVERFLOW2, iFreePage, ref rc);
                    if (rc != RC.OK)
                        return rc;
                }
            }
            // Fix the database pointer on page iPtrPage that pointed at iDbPage so that it points at iFreePage. Also fix the pointer map entry for iPtrPage.
            if (eType != PTRMAP.ROOTPAGE)
            {
                rc = pBt.btreeGetPage(iPtrPage, ref pPtrPage, 0);
                if (rc != RC.OK)
                    return rc;
                rc = Pager.sqlite3PagerWrite(pPtrPage.DbPage);
                if (rc != RC.OK)
                {
                    pPtrPage.releasePage();
                    return rc;
                }
                rc = pPtrPage.modifyPagePointer(iDbPage, iFreePage, eType);
                pPtrPage.releasePage();
                if (rc == RC.OK)
                    pBt.ptrmapPut(iFreePage, eType, iPtrPage, ref rc);
            }
            return rc;
        }

        internal static RC incrVacuumStep(BtShared pBt, Pgno nFin, Pgno iLastPg)
        {
            Pgno nFreeList;           // Number of pages still on the free-list
            Debug.Assert(MutexEx.Held(pBt.Mutex));
            Debug.Assert(iLastPg > nFin);
            if (!PTRMAP_ISPAGE(pBt, iLastPg) && iLastPg != PENDING_BYTE_PAGE(pBt))
            {
                PTRMAP eType = 0;
                Pgno iPtrPage = 0;
                nFreeList = ConvertEx.Get4(pBt.Page1.Data, 36);
                if (nFreeList == 0)
                    return RC.DONE;
                var rc = pBt.ptrmapGet( iLastPg, ref eType, ref iPtrPage);
                if (rc != RC.OK)
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
                        if (rc != RC.OK)
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
                    if (rc != RC.OK)
                        return rc;
                    // If nFin is zero, this loop runs exactly once and page pLastPg is swapped with the first free page pulled off the free list.
                    // On the other hand, if nFin is greater than zero, then keep looping until a free-page located within the first nFin pages of the file is found.
                    do
                    {
                        var pFreePg = new MemPage();
                        rc = pBt.allocateBtreePage(ref pFreePg, ref iFreePg, 0, 0);
                        if (rc != RC.OK)
                        {
                            pLastPg.releasePage();
                            return rc;
                        }
                        pFreePg.releasePage();
                    } while (nFin != 0 && iFreePg > nFin);
                    Debug.Assert(iFreePg < iLastPg);
                    rc = Pager.sqlite3PagerWrite(pLastPg.DbPage);
                    if (rc == RC.OK)
                        rc = relocatePage(pBt, pLastPg, eType, iPtrPage, iFreePg, (nFin != 0) ? 1 : 0);
                    pLastPg.releasePage();
                    if (rc != RC.OK)
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
                        if (rc != RC.OK)
                            return rc;
                        rc = Pager.sqlite3PagerWrite(pPg.DbPage);
                        pPg.releasePage();
                        if (rc != RC.OK)
                            return rc;
                    }
                    iLastPg--;
                }
                pBt.Pager.sqlite3PagerTruncateImage(iLastPg);
                pBt.Pages = iLastPg;
            }
            return RC.OK;
        }

        internal static RC sqlite3BtreeIncrVacuum(Btree p)
        {
            var pBt = p.Shared;
            p.sqlite3BtreeEnter();
            Debug.Assert(pBt.InTransaction == TRANS.WRITE && p.InTransaction == TRANS.WRITE);
            RC rc;
            if (!pBt.AutoVacuum)
                rc = RC.DONE;
            else
            {
                Btree.invalidateAllOverflowCache(pBt);
                rc = incrVacuumStep(pBt, 0, pBt.btreePagecount());
                if (rc == RC.OK)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.Page1.DbPage);
                    ConvertEx.Put4(pBt.Page1.Data, 28, pBt.Pages);
                }
            }
            p.sqlite3BtreeLeave();
            return rc;
        }

        internal static RC autoVacuumCommit(BtShared pBt)
        {
            var rc = RC.OK;
            var pPager = pBt.Pager;
#if DEBUG
            var nRef = pPager.sqlite3PagerRefcount();
#else
            var nRef = 0;
#endif
            Debug.Assert(MutexEx.Held(pBt.Mutex));
            Btree.invalidateAllOverflowCache(pBt);
            Debug.Assert(pBt.AutoVacuum);
            if (!pBt.IncrVacuum)
            {
                var nOrig = pBt.btreePagecount(); // Database size before freeing
                if (PTRMAP_ISPAGE(pBt, nOrig) || nOrig == PENDING_BYTE_PAGE(pBt))
                    // It is not possible to create a database for which the final page is either a pointer-map page or the pending-byte page. If one
                    // is encountered, this indicates corruption.
                    return SysEx.SQLITE_CORRUPT_BKPT();
                var nFree = (Pgno)ConvertEx.Get4(pBt.Page1.Data, 36); // Number of pages on the freelist initially
                var nEntry = (int)pBt.UsableSize / 5; // Number of entries on one ptrmap page
                var nPtrmap = (Pgno)((nFree - nOrig + PTRMAP_PAGENO(pBt, nOrig) + (Pgno)nEntry) / nEntry); // Number of PtrMap pages to be freed 
                var nFin = nOrig - nFree - nPtrmap; // Number of pages in database after autovacuuming
                if (nOrig > PENDING_BYTE_PAGE(pBt) && nFin < PENDING_BYTE_PAGE(pBt))
                    nFin--;
                while (PTRMAP_ISPAGE(pBt, nFin) || nFin == PENDING_BYTE_PAGE(pBt))
                    nFin--;
                if (nFin > nOrig)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                for (var iFree = nOrig; iFree > nFin && rc == RC.OK; iFree--)
                    rc = incrVacuumStep(pBt, nFin, iFree);
                if ((rc == RC.DONE || rc == RC.OK) && nFree > 0)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.Page1.DbPage);
                    ConvertEx.Put4(pBt.Page1.Data, 32, 0);
                    ConvertEx.Put4(pBt.Page1.Data, 36, 0);
                    ConvertEx.Put4(pBt.Page1.Data, 28, nFin);
                    pBt.Pager.sqlite3PagerTruncateImage(nFin);
                    pBt.Pages = nFin;
                }
                if (rc != RC.OK)
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
