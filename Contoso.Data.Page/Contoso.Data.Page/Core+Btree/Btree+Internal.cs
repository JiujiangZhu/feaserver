using System;
using System.Diagnostics;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;
using SAVEPOINT = Contoso.Core.Pager.SAVEPOINT;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;
using LOCK = Contoso.Core.Btree.BtreeLock.LOCK;

namespace Contoso.Core
{
    public partial class Btree
    {
        // was:btreeEndTransaction
        internal void btreeEndTransaction()
        {
            var pBt = this.Shared;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            pBt.btreeClearHasContent();
            if (this.InTransaction > TRANS.NONE && this.DB.activeVdbeCnt > 1)
            {
                // If there are other active statements that belong to this database handle, downgrade to a read-only transaction. The other statements
                // may still be reading from the database.
                downgradeAllSharedCacheTableLocks();
                this.InTransaction = TRANS.READ;
            }
            else
            {
                // If the handle had any kind of transaction open, decrement the transaction count of the shared btree. If the transaction count
                // reaches 0, set the shared state to TRANS_NONE. The unlockBtreeIfUnused() call below will unlock the pager.  */
                if (this.InTransaction != TRANS.NONE)
                {
                    clearAllSharedCacheTableLocks();
                    pBt.Transactions--;
                    if (pBt.Transactions == 0)
                        pBt.InTransaction = TRANS.NONE;
                }
                // Set the current transaction state to TRANS_NONE and unlock the pager if this call closed the only read or write transaction.
                this.InTransaction = TRANS.NONE;
                pBt.unlockBtreeIfUnused();
            }
            btreeIntegrity();
        }

        internal static void pageReinit(DbPage pData)
        {
            var pPage = Pager.sqlite3PagerGetExtra(pData);
            Debug.Assert(Pager.sqlite3PagerPageRefcount(pData) > 0);
            if (pPage.HasInit)
            {
                Debug.Assert(MutexEx.Held(pPage.Shared.Mutex));
                pPage.HasInit = false;
                if (Pager.sqlite3PagerPageRefcount(pData) > 1)
                    // pPage might not be a btree page;  it might be an overflow page or ptrmap page or a free page.  In those cases, the following
                    // call to btreeInitPage() will likely return SQLITE_CORRUPT. But no harm is done by this.  And it is very important that
                    // btreeInitPage() be called on every btree page so we make the call for every page that comes in for re-initing.
                    pPage.btreeInitPage();
            }
        }

        internal static int btreeInvokeBusyHandler(object pArg)
        {
            var pBt = (BtShared)pArg;
            Debug.Assert(pBt.DB != null);
            Debug.Assert(MutexEx.Held(pBt.DB.Mutex));
            return pBt.DB.sqlite3InvokeBusyHandler();
        }

        internal RC btreeCursor(int iTable, bool wrFlag, KeyInfo pKeyInfo, BtreeCursor pCur)
        {
            Debug.Assert(sqlite3BtreeHoldsMutex());
            // The following Debug.Assert statements verify that if this is a sharable b-tree database, the connection is holding the required table locks,
            // and that no other connection has any open cursor that conflicts with this lock.
            Debug.Assert(hasSharedCacheTableLock((uint)iTable, (pKeyInfo != null), (wrFlag ? LOCK.READ : LOCK.WRITE)));
            Debug.Assert(!wrFlag || !hasReadConflicts((uint)iTable));
            // Assert that the caller has opened the required transaction.
            Debug.Assert(this.InTransaction > TRANS.NONE);
            Debug.Assert(!wrFlag || this.InTransaction == TRANS.WRITE);
            var pBt = this.Shared;                 // Shared b-tree handle
            Debug.Assert(pBt.Page1 != null && pBt.Page1.Data != null);
            if (Check.NEVER(wrFlag && pBt.ReadOnly))
                return RC.READONLY;
            if (iTable == 1 && pBt.btreePagecount() == 0)
                return RC.EMPTY;
            // Now that no other errors can occur, finish filling in the BtCursor variables and link the cursor into the BtShared list.
            pCur.RootID = (Pgno)iTable;
            pCur.PageID = -1;
            pCur.KeyInfo = pKeyInfo;
            pCur.Tree = this;
            pCur.Shared = pBt;
            pCur.Writeable = wrFlag;
            pCur.Next = pBt.Cursors;
            if (pCur.Next != null)
                pCur.Next.Prev = pCur;
            pBt.Cursors = pCur;
            pCur.State = CURSOR.INVALID;
            pCur._cachedRowID = 0;
            return RC.OK;
        }

        internal RC btreeCreateTable(ref int piTable, CREATETABLE createTabFlags)
        {
            var pBt = this.Shared;
            var pRoot = new MemPage();
            Pgno pgnoRoot = 0;
            int ptfFlags;          // Page-type flage for the root page of new table
            RC rc;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(pBt.InTransaction == TRANS.WRITE);
            Debug.Assert(!pBt.ReadOnly);
#if SQLITE_OMIT_AUTOVACUUM
            rc = allocateBtreePage(pBt, ref pRoot, ref pgnoRoot, 1, 0);
            if (rc != SQLITE.OK)
                return rc;
#else
            if (pBt.AutoVacuum)
            {
                Pgno pgnoMove = 0; // Move a page here to make room for the root-page
                var pPageMove = new MemPage(); // The page to move to.
                // Creating a new table may probably require moving an existing database to make room for the new tables root page. In case this page turns
                // out to be an overflow page, delete all overflow page-map caches held by open cursors.
                invalidateAllOverflowCache(pBt);
                // Read the value of meta[3] from the database to determine where the root page of the new table should go. meta[3] is the largest root-page
                // created so far, so the new root-page is (meta[3]+1).
                GetMeta((int)META.LARGEST_ROOT_PAGE, ref pgnoRoot);
                pgnoRoot++;
                // The new root-page may not be allocated on a pointer-map page, or the PENDING_BYTE page.
                while (pgnoRoot == MemPage.PTRMAP_PAGENO(pBt, pgnoRoot) || pgnoRoot == MemPage.PENDING_BYTE_PAGE(pBt))
                    pgnoRoot++;
                Debug.Assert(pgnoRoot >= 3);
                // Allocate a page. The page that currently resides at pgnoRoot will be moved to the allocated page (unless the allocated page happens to reside at pgnoRoot).
                rc = pBt.allocateBtreePage(ref pPageMove, ref pgnoMove, pgnoRoot, 1);
                if (rc != RC.OK)
                    return rc;
                if (pgnoMove != pgnoRoot)
                {
                    // pgnoRoot is the page that will be used for the root-page of the new table (assuming an error did not occur). But we were
                    // allocated pgnoMove. If required (i.e. if it was not allocated by extending the file), the current page at position pgnoMove
                    // is already journaled.
                    PTRMAP eType = 0;
                    Pgno iPtrPage = 0;
                    pPageMove.releasePage();
                    // Move the page currently at pgnoRoot to pgnoMove.
                    rc = pBt.btreeGetPage(pgnoRoot, ref pRoot, 0);
                    if (rc != RC.OK)
                        return rc;
                    rc = pBt.ptrmapGet(pgnoRoot, ref eType, ref iPtrPage);
                    if (eType == PTRMAP.ROOTPAGE || eType == PTRMAP.FREEPAGE)
                        rc = SysEx.SQLITE_CORRUPT_BKPT();
                    if (rc != RC.OK)
                    {
                        pRoot.releasePage();
                        return rc;
                    }
                    Debug.Assert(eType != PTRMAP.ROOTPAGE);
                    Debug.Assert(eType != PTRMAP.FREEPAGE);
                    rc = MemPage.relocatePage(pBt, pRoot, eType, iPtrPage, pgnoMove, 0);
                    pRoot.releasePage();
                    // Obtain the page at pgnoRoot
                    if (rc != RC.OK)
                        return rc;
                    rc = pBt.btreeGetPage(pgnoRoot, ref pRoot, 0);
                    if (rc != RC.OK)
                        return rc;
                    rc = Pager.sqlite3PagerWrite(pRoot.DbPage);
                    if (rc != RC.OK)
                    {
                        pRoot.releasePage();
                        return rc;
                    }
                }
                else
                    pRoot = pPageMove;
                // Update the pointer-map and meta-data with the new root-page number.
                pBt.ptrmapPut(pgnoRoot, PTRMAP.ROOTPAGE, 0, ref rc);
                if (rc != RC.OK)
                {
                    pRoot.releasePage();
                    return rc;
                }
                // When the new root page was allocated, page 1 was made writable in order either to increase the database filesize, or to decrement the
                // freelist count.  Hence, the sqlite3BtreeUpdateMeta() call cannot fail.
                Debug.Assert(Pager.sqlite3PagerIswriteable(pBt.Page1.DbPage));
                rc = SetMeta(4, pgnoRoot);
                if (Check.NEVER(rc != RC.OK))
                {
                    pRoot.releasePage();
                    return rc;
                }
            }
            else
            {
                rc = pBt.allocateBtreePage(ref pRoot, ref pgnoRoot, 1, 0);
                if (rc != RC.OK)
                    return rc;
            }
#endif
            Debug.Assert(Pager.sqlite3PagerIswriteable(pRoot.DbPage));
            ptfFlags = ((createTabFlags & CREATETABLE.INTKEY) != 0 ? PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF : PTF_ZERODATA | PTF_LEAF);
            pRoot.zeroPage(ptfFlags);
            Pager.sqlite3PagerUnref(pRoot.DbPage);
            Debug.Assert((pBt.OpenFlags & OPEN.SINGLE) == 0 || pgnoRoot == 2);
            piTable = (int)pgnoRoot;
            return RC.OK;
        }

        internal RC btreeDropTable(Pgno iTable, ref int piMoved)
        {
            MemPage pPage = null;
            var pBt = this.Shared;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(this.InTransaction == TRANS.WRITE);
            // It is illegal to drop a table if any cursors are open on the database. This is because in auto-vacuum mode the backend may
            // need to move another root-page to fill a gap left by the deleted root page. If an open cursor was using this page a problem would occur.
            // This error is caught long before control reaches this point.
            if (Check.NEVER(pBt.Cursors))
            {
                sqlite3.sqlite3ConnectionBlocked(this.DB, pBt.Cursors.Tree.DB);
                return RC.LOCKED_SHAREDCACHE;
            }
            var rc = pBt.btreeGetPage((Pgno)iTable, ref pPage, 0);
            if (rc != RC.OK)
                return rc;
            var dummy0 = 0;
            rc = ClearTable((int)iTable, ref dummy0);
            if (rc != RC.OK)
            {
                pPage.releasePage();
                return rc;
            }
            piMoved = 0;
            if (iTable > 1)
            {
#if SQLITE_OMIT_AUTOVACUUM
freePage(pPage, ref rc);
releasePage(pPage);
#else
                if (pBt.AutoVacuum)
                {
                    Pgno maxRootPgno = 0;
                    GetMeta((int)META.LARGEST_ROOT_PAGE, ref maxRootPgno);
                    if (iTable == maxRootPgno)
                    {
                        // If the table being dropped is the table with the largest root-page number in the database, put the root page on the free list.
                        pPage.freePage(ref rc);
                        pPage.releasePage();
                        if (rc != RC.OK)
                            return rc;
                    }
                    else
                    {
                        // The table being dropped does not have the largest root-page number in the database. So move the page that does into the
                        // gap left by the deleted root-page.
                        var pMove = new MemPage();
                        pPage.releasePage();
                        rc = pBt.btreeGetPage(maxRootPgno, ref pMove, 0);
                        if (rc != RC.OK)
                            return rc;
                        rc = MemPage.relocatePage(pBt, pMove, PTRMAP.ROOTPAGE, 0, iTable, 0);
                        pMove.releasePage();
                        if (rc != RC.OK)
                            return rc;
                        pMove = null;
                        rc = pBt.btreeGetPage(maxRootPgno, ref pMove, 0);
                        pMove.freePage(ref rc);
                        pMove.releasePage();
                        if (rc != RC.OK)
                            return rc;
                        piMoved = (int)maxRootPgno;
                    }
                    // Set the new 'max-root-page' value in the database header. This is the old value less one, less one more if that happens to
                    // be a root-page number, less one again if that is the PENDING_BYTE_PAGE.
                    maxRootPgno--;
                    while (maxRootPgno == MemPage.PENDING_BYTE_PAGE(pBt) || MemPage.PTRMAP_ISPAGE(pBt, maxRootPgno))
                        maxRootPgno--;
                    Debug.Assert(maxRootPgno != MemPage.PENDING_BYTE_PAGE(pBt));
                    rc = SetMeta(4, maxRootPgno);
                }
                else
                {
                    pPage.freePage(ref rc);
                    pPage.releasePage();
                }
#endif
            }
            else
            {
                // If sqlite3BtreeDropTable was called on page 1. This really never should happen except in a corrupt database.
                pPage.zeroPage(PTF_INTKEY | PTF_LEAF);
                pPage.releasePage();
            }
            return rc;
        }
    }
}
