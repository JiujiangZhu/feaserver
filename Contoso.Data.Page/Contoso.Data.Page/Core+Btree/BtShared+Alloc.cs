using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;

namespace Contoso.Core
{
    public partial class BtShared
    {
        internal SQLITE allocateBtreePage(ref MemPage ppPage, ref Pgno pPgno, Pgno nearby, byte exact)
        {
            MemPage pTrunk = null;
            MemPage pPrevTrunk = null;
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            var pPage1 = this.pPage1;
            var mxPage = btreePagecount(); // Total size of the database file
            var n = ConvertEx.sqlite3Get4byte(pPage1.aData, 36); // Number of pages on the freelist
            if (n >= mxPage)
                return SysEx.SQLITE_CORRUPT_BKPT();
            SQLITE rc;
            if (n > 0)
            {
                // There are pages on the freelist.  Reuse one of those pages.
                Pgno iTrunk;
                byte searchList = 0; // If the free-list must be searched for 'nearby'
                // If the 'exact' parameter was true and a query of the pointer-map shows that the page 'nearby' is somewhere on the free-list, then the entire-list will be searched for that page.
#if !SQLITE_OMIT_AUTOVACUUM
                if (exact != 0 && nearby <= mxPage)
                {
                    Debug.Assert(nearby > 0);
                    Debug.Assert(this.autoVacuum);
                    PTRMAP eType = 0;
                    uint dummy0 = 0;
                    rc = ptrmapGet(nearby, ref eType, ref dummy0);
                    if (rc != SQLITE.OK)
                        return rc;
                    if (eType == PTRMAP.FREEPAGE)
                        searchList = 1;
                    pPgno = nearby;
                }
#endif
                // Decrement the free-list count by 1. Set iTrunk to the index of the first free-list trunk page. iPrevTrunk is initially 1.
                rc = Pager.sqlite3PagerWrite(pPage1.pDbPage);
                if (rc != SQLITE.OK)
                    return rc;
                ConvertEx.sqlite3Put4byte(pPage1.aData, (uint)36, n - 1);
                // The code within this loop is run only once if the 'searchList' variable is not true. Otherwise, it runs once for each trunk-page on the
                // free-list until the page 'nearby' is located.
                do
                {
                    pPrevTrunk = pTrunk;
                    iTrunk = (pPrevTrunk != null ? ConvertEx.sqlite3Get4byte(pPrevTrunk.aData, 0) : ConvertEx.sqlite3Get4byte(pPage1.aData, 32));
                    rc = (iTrunk > mxPage ? SysEx.SQLITE_CORRUPT_BKPT() : btreeGetPage(iTrunk, ref pTrunk, 0));
                    if (rc != SQLITE.OK)
                    {
                        pTrunk = null;
                        goto end_allocate_page;
                    }
                    var k = ConvertEx.sqlite3Get4byte(pTrunk.aData, 4); // # of leaves on this trunk page
                    if (k == 0 && searchList == 0)
                    {
                        // The trunk has no leaves and the list is not being searched. So extract the trunk page itself and use it as the newly allocated page
                        Debug.Assert(pPrevTrunk == null);
                        rc = Pager.sqlite3PagerWrite(pTrunk.pDbPage);
                        if (rc != SQLITE.OK)
                            goto end_allocate_page;
                        pPgno = iTrunk;
                        Buffer.BlockCopy(pTrunk.aData, 0, pPage1.aData, 32, 4);
                        ppPage = pTrunk;
                        pTrunk = null;
                        Btree.TRACE("ALLOCATE: %d trunk - %d free pages left\n", pPgno, n - 1);
                    }
                    else if (k > (uint)(this.usableSize / 4 - 2))
                    {
                        // Value of k is out of range. Database corruption
                        rc = SysEx.SQLITE_CORRUPT_BKPT();
                        goto end_allocate_page;
#if !SQLITE_OMIT_AUTOVACUUM
                    }
                    else if (searchList != 0 && nearby == iTrunk)
                    {
                        // The list is being searched and this trunk page is the page to allocate, regardless of whether it has leaves.
                        Debug.Assert(pPgno == iTrunk);
                        ppPage = pTrunk;
                        searchList = 0;
                        rc = Pager.sqlite3PagerWrite(pTrunk.pDbPage);
                        if (rc != SQLITE.OK)
                            goto end_allocate_page;
                        if (k == 0)
                        {
                            if (pPrevTrunk == null)
                            {
                                pPage1.aData[32 + 0] = pTrunk.aData[0 + 0];
                                pPage1.aData[32 + 1] = pTrunk.aData[0 + 1];
                                pPage1.aData[32 + 2] = pTrunk.aData[0 + 2];
                                pPage1.aData[32 + 3] = pTrunk.aData[0 + 3];
                            }
                            else
                            {
                                rc = Pager.sqlite3PagerWrite(pPrevTrunk.pDbPage);
                                if (rc != SQLITE.OK)
                                    goto end_allocate_page;
                                pPrevTrunk.aData[0 + 0] = pTrunk.aData[0 + 0];
                                pPrevTrunk.aData[0 + 1] = pTrunk.aData[0 + 1];
                                pPrevTrunk.aData[0 + 2] = pTrunk.aData[0 + 2];
                                pPrevTrunk.aData[0 + 3] = pTrunk.aData[0 + 3];
                            }
                        }
                        else
                        {
                            // The trunk page is required by the caller but it contains pointers to free-list leaves. The first leaf becomes a trunk page in this case.
                            var pNewTrunk = new MemPage();
                            var iNewTrunk = (Pgno)ConvertEx.sqlite3Get4byte(pTrunk.aData, 8);
                            if (iNewTrunk > mxPage)
                            {
                                rc = SysEx.SQLITE_CORRUPT_BKPT();
                                goto end_allocate_page;
                            }
                            rc = btreeGetPage(iNewTrunk, ref pNewTrunk, 0);
                            if (rc != SQLITE.OK)
                                goto end_allocate_page;
                            rc = Pager.sqlite3PagerWrite(pNewTrunk.pDbPage);
                            if (rc != SQLITE.OK)
                            {
                                pNewTrunk.releasePage();
                                goto end_allocate_page;
                            }
                            pNewTrunk.aData[0 + 0] = pTrunk.aData[0 + 0];
                            pNewTrunk.aData[0 + 1] = pTrunk.aData[0 + 1];
                            pNewTrunk.aData[0 + 2] = pTrunk.aData[0 + 2];
                            pNewTrunk.aData[0 + 3] = pTrunk.aData[0 + 3];
                            ConvertEx.sqlite3Put4byte(pNewTrunk.aData, (uint)4, (uint)(k - 1));
                            Buffer.BlockCopy(pTrunk.aData, 12, pNewTrunk.aData, 8, (int)(k - 1) * 4);
                            pNewTrunk.releasePage();
                            if (pPrevTrunk == null)
                            {
                                Debug.Assert(Pager.sqlite3PagerIswriteable(pPage1.pDbPage));
                                ConvertEx.sqlite3Put4byte(pPage1.aData, (uint)32, iNewTrunk);
                            }
                            else
                            {
                                rc = Pager.sqlite3PagerWrite(pPrevTrunk.pDbPage);
                                if (rc != SQLITE.OK)
                                    goto end_allocate_page;
                                ConvertEx.sqlite3Put4byte(pPrevTrunk.aData, (uint)0, iNewTrunk);
                            }
                        }
                        pTrunk = null;
                        Btree.TRACE("ALLOCATE: %d trunk - %d free pages left\n", pPgno, n - 1);
#endif
                    }
                    else if (k > 0)
                    {
                        // Extract a leaf from the trunk
                        uint closest;
                        var aData = pTrunk.aData;
                        if (nearby > 0)
                        {
                            closest = 0;
                            var dist = Math.Abs((int)(ConvertEx.sqlite3Get4byte(aData, 8) - nearby));
                            for (uint i = 1; i < k; i++)
                            {
                                int dist2 = Math.Abs((int)(ConvertEx.sqlite3Get4byte(aData, 8 + i * 4) - nearby));
                                if (dist2 < dist)
                                {
                                    closest = i;
                                    dist = dist2;
                                }
                            }
                        }
                        else
                            closest = 0;
                        //
                        var iPage = (Pgno)ConvertEx.sqlite3Get4byte(aData, 8 + closest * 4);
                        if (iPage > mxPage)
                        {
                            rc = SysEx.SQLITE_CORRUPT_BKPT();
                            goto end_allocate_page;
                        }
                        if (searchList == 0 || iPage == nearby)
                        {
                            pPgno = iPage;
                            Btree.TRACE("ALLOCATE: %d was leaf %d of %d on trunk %d" + ": %d more free pages\n", pPgno, closest + 1, k, pTrunk.pgno, n - 1);
                            rc = Pager.sqlite3PagerWrite(pTrunk.pDbPage);
                            if (rc != SQLITE.OK)
                                goto end_allocate_page;
                            if (closest < k - 1)
                                Buffer.BlockCopy(aData, (int)(4 + k * 4), aData, 8 + (int)closest * 4, 4);
                            ConvertEx.sqlite3Put4byte(aData, (uint)4, (k - 1));
                            var noContent = (!btreeGetHasContent(pPgno) ? 1 : 0);
                            rc = btreeGetPage(pPgno, ref ppPage, noContent);
                            if (rc == SQLITE.OK)
                            {
                                rc = Pager.sqlite3PagerWrite((ppPage).pDbPage);
                                if (rc != SQLITE.OK)
                                    ppPage.releasePage();
                            }
                            searchList = 0;
                        }
                    }
                    pPrevTrunk.releasePage();
                    pPrevTrunk = null;
                } while (searchList != 0);
            }
            else
            {
                // There are no pages on the freelist, so create a new page at the end of the file
                rc = Pager.sqlite3PagerWrite(this.pPage1.pDbPage);
                if (rc != SQLITE.OK)
                    return rc;
                this.nPage++;
                if (this.nPage == MemPage.PENDING_BYTE_PAGE(this))
                    this.nPage++;
#if !SQLITE_OMIT_AUTOVACUUM
                if (this.autoVacuum && MemPage.PTRMAP_ISPAGE(this, this.nPage))
                {
                    // If pPgno refers to a pointer-map page, allocate two new pages at the end of the file instead of one. The first allocated page
                    // becomes a new pointer-map page, the second is used by the caller.
                    MemPage pPg = null;
                    Btree.TRACE("ALLOCATE: %d from end of file (pointer-map page)\n", pPgno);
                    Debug.Assert(this.nPage != MemPage.PENDING_BYTE_PAGE(this));
                    rc = btreeGetPage(this.nPage, ref pPg, 1);
                    if (rc == SQLITE.OK)
                    {
                        rc = Pager.sqlite3PagerWrite(pPg.pDbPage);
                        pPg.releasePage();
                    }
                    if (rc != SQLITE.OK)
                        return rc;
                    this.nPage++;
                    if (this.nPage == MemPage.PENDING_BYTE_PAGE(this))
                        this.nPage++;
                }
#endif
                ConvertEx.sqlite3Put4byte(this.pPage1.aData, (uint)28, this.nPage);
                pPgno = this.nPage;
                Debug.Assert(pPgno != MemPage.PENDING_BYTE_PAGE(this));
                rc = btreeGetPage(pPgno, ref ppPage, 1);
                if (rc != SQLITE.OK)
                    return rc;
                rc = Pager.sqlite3PagerWrite((ppPage).pDbPage);
                if (rc != SQLITE.OK)
                    ppPage.releasePage();
                Btree.TRACE("ALLOCATE: %d from end of file\n", pPgno);
            }
            Debug.Assert(pPgno != MemPage.PENDING_BYTE_PAGE(this));

        end_allocate_page:
            pTrunk.releasePage();
            pPrevTrunk.releasePage();
            if (rc == SQLITE.OK)
            {
                if (Pager.sqlite3PagerPageRefcount((ppPage).pDbPage) > 1)
                {
                    ppPage.releasePage();
                    return SysEx.SQLITE_CORRUPT_BKPT();
                }
                (ppPage).isInit = 0;
            }
            else
                ppPage = null;
            Debug.Assert(rc != SQLITE.OK || Pager.sqlite3PagerIswriteable((ppPage).pDbPage));
            return rc;
        }

        internal SQLITE freePage2(MemPage pMemPage, Pgno iPage)
        {
            MemPage pTrunk = null; // Free-list trunk page
            var pPage1 = this.pPage1; // Local reference to page 1
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.mutex));
            Debug.Assert(iPage > 1);
            Debug.Assert(pMemPage == null || pMemPage.pgno == iPage);
            MemPage pPage; // Page being freed. May be NULL.
            if (pMemPage != null)
            {
                pPage = pMemPage;
                Pager.sqlite3PagerRef(pPage.pDbPage);
            }
            else
                pPage = btreePageLookup(iPage);
            // Increment the free page count on pPage1
            var rc = Pager.sqlite3PagerWrite(pPage1.pDbPage);
            if (rc != SQLITE.OK)
                goto freepage_out;
            var nFree = (int)ConvertEx.sqlite3Get4byte(pPage1.aData, 36); // Initial number of pages on free-list
            ConvertEx.sqlite3Put4byte(pPage1.aData, 36, nFree + 1);
            if (this.secureDelete)
            {
                // If the secure_delete option is enabled, then always fully overwrite deleted information with zeros.
                if ((pPage == null && ((rc = btreeGetPage(iPage, ref pPage, 0)) != SQLITE.OK)) || ((rc = Pager.sqlite3PagerWrite(pPage.pDbPage)) != SQLITE.OK))
                    goto freepage_out;
                Array.Clear(pPage.aData, 0, (int)pPage.pBt.pageSize);
            }
            // If the database supports auto-vacuum, write an entry in the pointer-map to indicate that the page is free.
#if !SQLITE_OMIT_AUTOVACUUM
            if (this.autoVacuum)
#else
if (false)
#endif
            {
                ptrmapPut(iPage, PTRMAP.FREEPAGE, 0, ref rc);
                if (rc != SQLITE.OK)
                    goto freepage_out;
            }
            // Now manipulate the actual database free-list structure. There are two possibilities. If the free-list is currently empty, or if the first
            // trunk page in the free-list is full, then this page will become a new free-list trunk page. Otherwise, it will become a leaf of the
            // first trunk page in the current free-list. This block tests if it is possible to add the page as a new free-list leaf.
            Pgno iTrunk = 0; // Page number of free-list trunk page
            if (nFree != 0)
            {
                uint nLeaf; // Initial number of leaf cells on trunk page
                iTrunk = (Pgno)ConvertEx.sqlite3Get4byte(pPage1.aData, 32); // Page number of free-list trunk page 
                rc = btreeGetPage(iTrunk, ref pTrunk, 0);
                if (rc != SQLITE.OK)
                    goto freepage_out;
                nLeaf = ConvertEx.sqlite3Get4byte(pTrunk.aData, 4);
                Debug.Assert(this.usableSize > 32);
                if (nLeaf > (uint)this.usableSize / 4 - 2)
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto freepage_out;
                }
                if (nLeaf < (uint)this.usableSize / 4 - 8)
                {
                    // In this case there is room on the trunk page to insert the page being freed as a new leaf.
                    // Note: that the trunk page is not really full until it contains usableSize/4 - 2 entries, not usableSize/4 - 8 entries as we have
                    // coded.  But due to a coding error in versions of SQLite prior to 3.6.0, databases with freelist trunk pages holding more than
                    // usableSize/4 - 8 entries will be reported as corrupt.  In order to maintain backwards compatibility with older versions of SQLite,
                    // we will continue to restrict the number of entries to usableSize/4 - 8 for now.  At some point in the future (once everyone has upgraded
                    // to 3.6.0 or later) we should consider fixing the conditional above to read "usableSize/4-2" instead of "usableSize/4-8".
                    rc = Pager.sqlite3PagerWrite(pTrunk.pDbPage);
                    if (rc == SQLITE.OK)
                    {
                        ConvertEx.sqlite3Put4byte(pTrunk.aData, (uint)4, nLeaf + 1);
                        ConvertEx.sqlite3Put4byte(pTrunk.aData, (uint)8 + nLeaf * 4, iPage);
                        if (pPage != null && !this.secureDelete)
                            Pager.sqlite3PagerDontWrite(pPage.pDbPage);
                        rc = btreeSetHasContent(iPage);
                    }
                    Btree.TRACE("FREE-PAGE: %d leaf on trunk page %d\n", iPage, pTrunk.pgno);
                    goto freepage_out;
                }
            }
            // If control flows to this point, then it was not possible to add the the page being freed as a leaf page of the first trunk in the free-list.
            // Possibly because the free-list is empty, or possibly because the first trunk in the free-list is full. Either way, the page being freed
            // will become the new first trunk page in the free-list.
            if (pPage == null && (rc = btreeGetPage(iPage, ref pPage, 0)) != SQLITE.OK)
                goto freepage_out;
            rc = Pager.sqlite3PagerWrite(pPage.pDbPage);
            if (rc != SQLITE.OK)
                goto freepage_out;
            ConvertEx.sqlite3Put4byte(pPage.aData, iTrunk);
            ConvertEx.sqlite3Put4byte(pPage.aData, 4, 0);
            ConvertEx.sqlite3Put4byte(pPage1.aData, (uint)32, iPage);
            Btree.TRACE("FREE-PAGE: %d new trunk page replacing %d\n", pPage.pgno, iTrunk);
        freepage_out:
            if (pPage != null)
                pPage.isInit = 0;
            pPage.releasePage();
            pTrunk.releasePage();
            return rc;
        }
    }
}
