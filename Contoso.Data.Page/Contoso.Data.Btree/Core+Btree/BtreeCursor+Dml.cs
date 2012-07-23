using System;
using System.Diagnostics;
using TRANS = Contoso.Core.Btree.TRANS;
using LOCK = Contoso.Core.Btree.BtreeLock.LOCK;

namespace Contoso.Core
{
    public partial class BtreeCursor
    {
        private static readonly byte[] _balanceQuickSpaces = new byte[13];

        // was:balance
        private RC Balance()
        {
            var rc = RC.OK;
            var nMin = (int)this.Shared.UsableSize * 2 / 3;
            var balance_quick_called = 0;
            var balance_deeper_called = 0;
            do
            {
                var pageID = this.PageID;
                var page = this.Pages[pageID];
                if (pageID == 0)
                {
                    if (page.NOverflows != 0)
                    {
                        // The root page of the b-tree is overfull. In this case call the balance_deeper() function to create a new child for the root-page
                        // and copy the current contents of the root-page to it. The next iteration of the do-loop will balance the child page.
                        Debug.Assert((balance_deeper_called++) == 0);
                        rc = MemPage.balance_deeper(page, ref this.Pages[1]);
                        if (rc == RC.OK)
                        {
                            this.PageID = 1;
                            this.PagesIndexs[0] = 0;
                            this.PagesIndexs[1] = 0;
                            Debug.Assert(this.Pages[1].NOverflows != 0);
                        }
                    }
                    else
                        break;
                }
                else if (page.NOverflows == 0 && page.FreeBytes <= nMin)
                    break;
                else
                {
                    var pParent = this.Pages[pageID - 1];
                    var iIdx = this.PagesIndexs[pageID - 1];
                    rc = Pager.Write(pParent.DbPage);
                    if (rc == RC.OK)
                    {
#if !SQLITE_OMIT_QUICKBALANCE
                        if (page.HasData != 0 && page.NOverflows == 1 && page.Overflows[0].Index == page.Cells && pParent.ID != 1 && pParent.Cells == iIdx)
                        {
                            // Call balance_quick() to create a new sibling of pPage on which to store the overflow cell. balance_quick() inserts a new cell
                            // into pParent, which may cause pParent overflow. If this happens, the next interation of the do-loop will balance pParent
                            // use either balance_nonroot() or balance_deeper(). Until this happens, the overflow cell is stored in the aBalanceQuickSpace[]
                            // buffer.
                            // The purpose of the following Debug.Assert() is to check that only a single call to balance_quick() is made for each call to this
                            // function. If this were not verified, a subtle bug involving reuse of the aBalanceQuickSpace[] might sneak in.
                            Debug.Assert((balance_quick_called++) == 0);
                            rc = MemPage.balance_quick(pParent, page, _balanceQuickSpaces);
                        }
                        else
#endif
                        {
                            // In this case, call balance_nonroot() to redistribute cells between pPage and up to 2 of its sibling pages. This involves
                            // modifying the contents of pParent, which may cause pParent to become overfull or underfull. The next iteration of the do-loop
                            // will balance the parent page to correct this.
                            // If the parent page becomes overfull, the overflow cell or cells are stored in the pSpace buffer allocated immediately below.
                            // A subsequent iteration of the do-loop will deal with this by calling balance_nonroot() (balance_deeper() may be called first,
                            // but it doesn't deal with overflow cells - just moves them to a different page). Once this subsequent call to balance_nonroot()
                            // has completed, it is safe to release the pSpace buffer used by the previous call, as the overflow cell data will have been
                            // copied either into the body of a database page or into the new pSpace buffer passed to the latter call to balance_nonroot().
                            var pSpace = new byte[this.Shared.PageSize];// u8 pSpace = sqlite3PageMalloc( pCur.pBt.pageSize );
                            rc = MemPage.balance_nonroot(pParent, iIdx, null, pageID == 1 ? 1 : 0);
                            // The pSpace buffer will be freed after the next call to balance_nonroot(), or just before this function returns, whichever comes first.
                        }
                    }
                    page.NOverflows = 0;
                    // The next iteration of the do-loop balances the parent page.
                    page.releasePage();
                    this.PageID--;
                }
            } while (rc == RC.OK);
            return rc;
        }

        // was:sqlite3BtreeInsert
        public RC Insert(byte[] key, long nKey, byte[] data, int nData, int nZero, bool appendBiasRight, int seekResult)
        {
            var loc = seekResult; // -1: before desired location  +1: after
            var szNew = 0;
            int idx;
            var p = this.Tree;
            var pBt = p.Shared;
            int oldCell;
            byte[] newCell = null;
            if (this.State == CursorState.FAULT)
            {
                Debug.Assert(this.SkipNext != 0);
                return (RC)this.SkipNext;
            }
            Debug.Assert(HoldsMutex());
            Debug.Assert(this.Writeable && pBt.InTransaction == TRANS.WRITE && !pBt.ReadOnly);
            Debug.Assert(p.hasSharedCacheTableLock(this.RootID, (this.KeyInfo != null), LOCK.WRITE));
            // Assert that the caller has been consistent. If this cursor was opened expecting an index b-tree, then the caller should be inserting blob
            // keys with no associated data. If the cursor was opened expecting an intkey table, the caller should be inserting integer keys with a
            // blob of associated data.
            Debug.Assert((key == null) == (this.KeyInfo == null));
            // If this is an insert into a table b-tree, invalidate any incrblob cursors open on the row being replaced (assuming this is a replace
            // operation - if it is not, the following is a no-op).  */
            if (this.KeyInfo == null)
                Btree.invalidateIncrblobCursors(p, nKey, false);
            // Save the positions of any other cursors open on this table.
            // In some cases, the call to btreeMoveto() below is a no-op. For example, when inserting data into a table with auto-generated integer
            // keys, the VDBE layer invokes sqlite3BtreeLast() to figure out the integer key to use. It then calls this function to actually insert the
            // data into the intkey B-Tree. In this case btreeMoveto() recognizes that the cursor is already where it needs to be and returns without
            // doing any work. To avoid thwarting these optimizations, it is important not to clear the cursor here.
            var rc = pBt.saveAllCursors(this.RootID, this);
            if (rc != RC.OK)
                return rc;
            if (loc == 0)
            {
                rc = BtreeMoveTo(key, nKey, appendBiasRight, ref loc);
                if (rc != RC.OK)
                    return rc;
            }
            Debug.Assert(this.State == CursorState.VALID || (this.State == CursorState.INVALID && loc != 0));
            var pPage = this.Pages[this.PageID];
            Debug.Assert(pPage.HasIntKey || nKey >= 0);
            Debug.Assert(pPage.Leaf != 0 || !pPage.HasIntKey);
            Btree.TRACE("INSERT: table=%d nkey=%lld ndata=%d page=%d %s\n", this.RootID, nKey, nData, pPage.ID, (loc == 0 ? "overwrite" : "new entry"));
            Debug.Assert(pPage.HasInit);
            pBt.allocateTempSpace();
            newCell = pBt.pTmpSpace;
            rc = pPage.fillInCell(newCell, key, nKey, data, nData, nZero, ref szNew);
            if (rc != RC.OK)
                goto end_insert;
            Debug.Assert(szNew == pPage.cellSizePtr(newCell));
            Debug.Assert(szNew <= Btree.MX_CELL_SIZE(pBt));
            idx = this.PagesIndexs[this.PageID];
            if (loc == 0)
            {
                ushort szOld;
                Debug.Assert(idx < pPage.Cells);
                rc = Pager.Write(pPage.DbPage);
                if (rc != RC.OK)
                    goto end_insert;
                oldCell = pPage.FindCell(idx);
                if (0 == pPage.Leaf)
                {
                    newCell[0] = pPage.Data[oldCell + 0];
                    newCell[1] = pPage.Data[oldCell + 1];
                    newCell[2] = pPage.Data[oldCell + 2];
                    newCell[3] = pPage.Data[oldCell + 3];
                }
                szOld = pPage.cellSizePtr(oldCell);
                rc = pPage.clearCell(oldCell);
                pPage.dropCell(idx, szOld, ref rc);
                if (rc != RC.OK)
                    goto end_insert;
            }
            else if (loc < 0 && pPage.Cells > 0)
            {
                Debug.Assert(pPage.Leaf != 0);
                idx = ++this.PagesIndexs[this.PageID];
            }
            else
                Debug.Assert(pPage.Leaf != 0);
            pPage.insertCell(idx, newCell, szNew, null, 0, ref rc);
            Debug.Assert(rc != RC.OK || pPage.Cells > 0 || pPage.NOverflows > 0);
            // If no error has occured and pPage has an overflow cell, call balance() to redistribute the cells within the tree. Since balance() may move
            // the cursor, zero the BtCursor.info.nSize and BtCursor.validNKey variables.
            // Previous versions of SQLite called moveToRoot() to move the cursor back to the root page as balance() used to invalidate the contents
            // of BtCursor.apPage[] and BtCursor.aiIdx[]. Instead of doing that, set the cursor state to "invalid". This makes common insert operations
            // slightly faster.
            // There is a subtle but important optimization here too. When inserting multiple records into an intkey b-tree using a single cursor (as can
            // happen while processing an "INSERT INTO ... SELECT" statement), it is advantageous to leave the cursor pointing to the last entry in
            // the b-tree if possible. If the cursor is left pointing to the last entry in the table, and the next row inserted has an integer key
            // larger than the largest existing key, it is possible to insert the row without seeking the cursor. This can be a big performance boost.
            this.Info.nSize = 0;
            this.ValidNKey = false;
            if (rc == RC.OK && pPage.NOverflows != 0)
            {
                rc = Balance();
                // Must make sure nOverflow is reset to zero even if the balance() fails. Internal data structure corruption will result otherwise.
                // Also, set the cursor state to invalid. This stops saveCursorPosition() from trying to save the current position of the cursor.
                this.Pages[this.PageID].NOverflows = 0;
                this.State = CursorState.INVALID;
            }
            Debug.Assert(this.Pages[this.PageID].NOverflows == 0);
        end_insert:
            return rc;
        }

        // was:sqlite3BtreeDelete
        public RC Delete()
        {
            MemPage pPage;                      // Page to delete cell from
            int pCell;                          // Pointer to cell to delete
            int iCellIdx;                       // Index of cell to delete
            int iCellDepth;                     // Depth of node containing pCell
            var p = this.Tree;
            var pBt = p.Shared;
            Debug.Assert(HoldsMutex());
            Debug.Assert(pBt.InTransaction == TRANS.WRITE);
            Debug.Assert(!pBt.ReadOnly);
            Debug.Assert(this.Writeable);
            Debug.Assert(p.hasSharedCacheTableLock(this.RootID, (this.KeyInfo != null), LOCK.WRITE));
            Debug.Assert(!p.hasReadConflicts(this.RootID));
            if (Check.NEVER(this.PagesIndexs[this.PageID] >= this.Pages[this.PageID].Cells) || Check.NEVER(this.State != CursorState.VALID))
                return RC.ERROR;
            // If this is a delete operation to remove a row from a table b-tree, invalidate any incrblob cursors open on the row being deleted.
            if (this.KeyInfo == null)
                Btree.invalidateIncrblobCursors(p, this.Info.nKey, false);
            iCellDepth = this.PageID;
            iCellIdx = this.PagesIndexs[iCellDepth];
            pPage = this.Pages[iCellDepth];
            pCell = pPage.FindCell(iCellIdx);
            // If the page containing the entry to delete is not a leaf page, move the cursor to the largest entry in the tree that is smaller than
            // the entry being deleted. This cell will replace the cell being deleted from the internal node. The 'previous' entry is used for this instead
            // of the 'next' entry, as the previous entry is always a part of the sub-tree headed by the child page of the cell being deleted. This makes
            // balancing the tree following the delete operation easier.
            RC rc;
            if (pPage.Leaf == 0)
            {
                var notUsed = 0;
                rc = MovePrevious(ref notUsed);
                if (rc != RC.OK)
                    return rc;
            }
            // Save the positions of any other cursors open on this table before making any modifications. Make the page containing the entry to be
            // deleted writable. Then free any overflow pages associated with the entry and finally remove the cell itself from within the page.
            rc = pBt.saveAllCursors(this.RootID, this);
            if (rc != RC.OK)
                return rc;
            rc = Pager.Write(pPage.DbPage);
            if (rc != RC.OK)
                return rc;
            rc = pPage.clearCell(pCell);
            pPage.dropCell(iCellIdx, pPage.cellSizePtr(pCell), ref rc);
            if (rc != RC.OK)
                return rc;
            // If the cell deleted was not located on a leaf page, then the cursor is currently pointing to the largest entry in the sub-tree headed
            // by the child-page of the cell that was just deleted from an internal node. The cell from the leaf node needs to be moved to the internal
            // node to replace the deleted cell.
            if (pPage.Leaf == 0)
            {
                var pLeaf = this.Pages[this.PageID];
                int nCell;
                var n = this.Pages[iCellDepth + 1].ID;
                pCell = pLeaf.FindCell(pLeaf.Cells - 1);
                nCell = pLeaf.cellSizePtr(pCell);
                Debug.Assert(Btree.MX_CELL_SIZE(pBt) >= nCell);
                rc = Pager.Write(pLeaf.DbPage);
                var pNext_4 = MallocEx.sqlite3Malloc(nCell + 4);
                Buffer.BlockCopy(pLeaf.Data, pCell - 4, pNext_4, 0, nCell + 4);
                pPage.insertCell(iCellIdx, pNext_4, nCell + 4, null, n, ref rc);
                pLeaf.dropCell(pLeaf.Cells - 1, nCell, ref rc);
                if (rc != RC.OK)
                    return rc;
            }
            // Balance the tree. If the entry deleted was located on a leaf page, then the cursor still points to that page. In this case the first
            // call to balance() repairs the tree, and the if(...) condition is never true.
            //
            // Otherwise, if the entry deleted was on an internal node page, then pCur is pointing to the leaf page from which a cell was removed to
            // replace the cell deleted from the internal node. This is slightly tricky as the leaf node may be underfull, and the internal node may
            // be either under or overfull. In this case run the balancing algorithm on the leaf node first. If the balance proceeds far enough up the
            // tree that we can be sure that any problem in the internal node has been corrected, so be it. Otherwise, after balancing the leaf node,
            // walk the cursor up the tree to the internal node and balance it as well.
            rc = Balance();
            if (rc == RC.OK && this.PageID > iCellDepth)
            {
                while (this.PageID > iCellDepth)
                    this.Pages[this.PageID--].releasePage();
                rc = Balance();
            }
            if (rc == RC.OK)
                MoveToRoot();
            return rc;
        }
    }
}
