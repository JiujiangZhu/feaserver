using System;
using System.Diagnostics;
using Contoso.Sys;
using TRANS = Contoso.Core.Btree.TRANS;

namespace Contoso.Core
{
    public partial class BtCursor
    {
        internal SQLITE sqlite3BtreeInsert(byte[] pKey, long nKey, byte[] pData, int nData, int nZero, int appendBias, int seekResult)
        {
            var loc = seekResult; // -1: before desired location  +1: after
            var szNew = 0;
            int idx;
            var p = this.pBtree;
            var pBt = p.pBt;
            int oldCell;
            byte[] newCell = null;
            if (this.eState == CURSOR.FAULT)
            {
                Debug.Assert(this.skipNext != 0);
                return (SQLITE)this.skipNext;
            }
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.wrFlag != 0 && pBt.inTransaction == TRANS.WRITE && !pBt.readOnly);
            Debug.Assert(Btree.hasSharedCacheTableLock(p, this.pgnoRoot, this.pKeyInfo != null ? 1 : 0, 2));
            // Assert that the caller has been consistent. If this cursor was opened expecting an index b-tree, then the caller should be inserting blob
            // keys with no associated data. If the cursor was opened expecting an intkey table, the caller should be inserting integer keys with a
            // blob of associated data.
            Debug.Assert((pKey == null) == (this.pKeyInfo == null));
            // If this is an insert into a table b-tree, invalidate any incrblob cursors open on the row being replaced (assuming this is a replace
            // operation - if it is not, the following is a no-op).  */
            if (this.pKeyInfo == null)
                Btree.invalidateIncrblobCursors(p, nKey, 0);
            // Save the positions of any other cursors open on this table.
            // In some cases, the call to btreeMoveto() below is a no-op. For example, when inserting data into a table with auto-generated integer
            // keys, the VDBE layer invokes sqlite3BtreeLast() to figure out the integer key to use. It then calls this function to actually insert the
            // data into the intkey B-Tree. In this case btreeMoveto() recognizes that the cursor is already where it needs to be and returns without
            // doing any work. To avoid thwarting these optimizations, it is important not to clear the cursor here.
            var rc = pBt.saveAllCursors(this.pgnoRoot, this);
            if (rc != SQLITE.OK)
                return rc;
            if (loc == 0)
            {
                rc = btreeMoveto(pKey, nKey, appendBias, ref loc);
                if (rc != SQLITE.OK)
                    return rc;
            }
            Debug.Assert(this.eState == CURSOR.VALID || (this.eState == CURSOR.INVALID && loc != 0));
            var pPage = this.apPage[this.iPage];
            Debug.Assert(pPage.intKey != 0 || nKey >= 0);
            Debug.Assert(pPage.leaf != 0 || 0 == pPage.intKey);
            Btree.TRACE("INSERT: table=%d nkey=%lld ndata=%d page=%d %s\n", this.pgnoRoot, nKey, nData, pPage.pgno, (loc == 0 ? "overwrite" : "new entry"));
            Debug.Assert(pPage.isInit != 0);
            pBt.allocateTempSpace();
            newCell = pBt.pTmpSpace;
            rc = pPage.fillInCell(newCell, pKey, nKey, pData, nData, nZero, ref szNew);
            if (rc != SQLITE.OK)
                goto end_insert;
            Debug.Assert(szNew == pPage.cellSizePtr(newCell));
            Debug.Assert(szNew <= Btree.MX_CELL_SIZE(pBt));
            idx = this.aiIdx[this.iPage];
            if (loc == 0)
            {
                ushort szOld;
                Debug.Assert(idx < pPage.nCell);
                rc = Pager.sqlite3PagerWrite(pPage.pDbPage);
                if (rc != SQLITE.OK)
                    goto end_insert;
                oldCell = pPage.findCell(idx);
                if (0 == pPage.leaf)
                {
                    newCell[0] = pPage.aData[oldCell + 0];
                    newCell[1] = pPage.aData[oldCell + 1];
                    newCell[2] = pPage.aData[oldCell + 2];
                    newCell[3] = pPage.aData[oldCell + 3];
                }
                szOld = pPage.cellSizePtr(oldCell);
                rc = pPage.clearCell(oldCell);
                pPage.dropCell(idx, szOld, ref rc);
                if (rc != SQLITE.OK)
                    goto end_insert;
            }
            else if (loc < 0 && pPage.nCell > 0)
            {
                Debug.Assert(pPage.leaf != 0);
                idx = ++this.aiIdx[this.iPage];
            }
            else
                Debug.Assert(pPage.leaf != 0);
            pPage.insertCell(idx, newCell, szNew, null, 0, ref rc);
            Debug.Assert(rc != SQLITE.OK || pPage.nCell > 0 || pPage.nOverflow > 0);
            // If no error has occured and pPage has an overflow cell, call balance() to redistribute the cells within the tree. Since balance() may move
            // the cursor, zero the BtCursor.info.nSize and BtCursor.validNKey variables.
            // Previous versions of SQLite called moveToRoot() to move the cursor back to the root page as balance() used to invalidate the contents
            // of BtCursor.apPage[] and BtCursor.aiIdx[]. Instead of doing that, set the cursor state to "invalid". This makes common insert operations
            // slightly faster.
            // There is a subtle but important optimization here too. When inserting multiple records into an intkey b-tree using a single cursor (as can
            // happen while processing an "INSERT INTO ... SELECT" statement), it is advantageous to leave the cursor pointing to the last entry in
            // the b-tree if possible. If the cursor is left pointing to the last entry in the table, and the next row inserted has an integer key
            // larger than the largest existing key, it is possible to insert the row without seeking the cursor. This can be a big performance boost.
            this.info.nSize = 0;
            this.validNKey = false;
            if (rc == SQLITE.OK && pPage.nOverflow != 0)
            {
                rc = balance();
                // Must make sure nOverflow is reset to zero even if the balance() fails. Internal data structure corruption will result otherwise.
                // Also, set the cursor state to invalid. This stops saveCursorPosition() from trying to save the current position of the cursor.
                this.apPage[this.iPage].nOverflow = 0;
                this.eState = CURSOR.INVALID;
            }
            Debug.Assert(this.apPage[this.iPage].nOverflow == 0);
        end_insert:
            return rc;
        }

        internal SQLITE sqlite3BtreeDelete()
        {
            MemPage pPage;                      // Page to delete cell from
            int pCell;                          // Pointer to cell to delete
            int iCellIdx;                       // Index of cell to delete
            int iCellDepth;                     // Depth of node containing pCell
            var p = this.pBtree;
            var pBt = p.pBt;
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(pBt.inTransaction == TRANS.WRITE);
            Debug.Assert(!pBt.readOnly);
            Debug.Assert(this.wrFlag != 0);
            Debug.Assert(Btree.hasSharedCacheTableLock(p, this.pgnoRoot, (this.pKeyInfo != null ? 1 : 0), 2));
            Debug.Assert(!Btree.hasReadConflicts(p, this.pgnoRoot));
            if (Check.NEVER(this.aiIdx[this.iPage] >= this.apPage[this.iPage].nCell) || Check.NEVER(this.eState != CURSOR.VALID))
                return SQLITE.ERROR;
            // If this is a delete operation to remove a row from a table b-tree, invalidate any incrblob cursors open on the row being deleted.
            if (this.pKeyInfo == null)
                Btree.invalidateIncrblobCursors(p, this.info.nKey, 0);
            iCellDepth = this.iPage;
            iCellIdx = this.aiIdx[iCellDepth];
            pPage = this.apPage[iCellDepth];
            pCell = pPage.findCell(iCellIdx);
            // If the page containing the entry to delete is not a leaf page, move the cursor to the largest entry in the tree that is smaller than
            // the entry being deleted. This cell will replace the cell being deleted from the internal node. The 'previous' entry is used for this instead
            // of the 'next' entry, as the previous entry is always a part of the sub-tree headed by the child page of the cell being deleted. This makes
            // balancing the tree following the delete operation easier.
            SQLITE rc;
            if (pPage.leaf == 0)
            {
                var notUsed = 0;
                rc = sqlite3BtreePrevious(ref notUsed);
                if (rc != SQLITE.OK)
                    return rc;
            }
            // Save the positions of any other cursors open on this table before making any modifications. Make the page containing the entry to be
            // deleted writable. Then free any overflow pages associated with the entry and finally remove the cell itself from within the page.
            rc = pBt.saveAllCursors(this.pgnoRoot, this);
            if (rc != SQLITE.OK)
                return rc;
            rc = Pager.sqlite3PagerWrite(pPage.pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            rc = pPage.clearCell(pCell);
            pPage.dropCell(iCellIdx, pPage.cellSizePtr(pCell), ref rc);
            if (rc != SQLITE.OK)
                return rc;
            // If the cell deleted was not located on a leaf page, then the cursor is currently pointing to the largest entry in the sub-tree headed
            // by the child-page of the cell that was just deleted from an internal node. The cell from the leaf node needs to be moved to the internal
            // node to replace the deleted cell.
            if (pPage.leaf == 0)
            {
                var pLeaf = this.apPage[this.iPage];
                int nCell;
                var n = this.apPage[iCellDepth + 1].pgno;
                pCell = pLeaf.findCell(pLeaf.nCell - 1);
                nCell = pLeaf.cellSizePtr(pCell);
                Debug.Assert(Btree.MX_CELL_SIZE(pBt) >= nCell);
                rc = Pager.sqlite3PagerWrite(pLeaf.pDbPage);
                var pNext_4 = MallocEx.sqlite3Malloc(nCell + 4);
                Buffer.BlockCopy(pLeaf.aData, pCell - 4, pNext_4, 0, nCell + 4);
                pPage.insertCell(iCellIdx, pNext_4, nCell + 4, null, n, ref rc);
                pLeaf.dropCell(pLeaf.nCell - 1, nCell, ref rc);
                if (rc != SQLITE.OK)
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
            rc = balance();
            if (rc == SQLITE.OK && this.iPage > iCellDepth)
            {
                while (this.iPage > iCellDepth)
                    this.apPage[this.iPage--].releasePage();
                rc = balance();
            }
            if (rc == SQLITE.OK)
                moveToRoot();
            return rc;
        }
    }
}
