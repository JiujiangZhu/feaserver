using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class BtreeCursor
    {
        // was:sqlite3BtreeClearCursor
        public void Clear()
        {
            Debug.Assert(HoldsMutex());
            MallocEx.sqlite3_free(ref this.Key);
            this.State = CursorState.INVALID;
        }

        // was:sqlite3BtreeCursorHasMoved
        public RC HasMoved(ref bool hasMoved)
        {
            var rc = RestorePosition();
            if (rc != RC.OK)
            {
                hasMoved = true;
                return rc;
            }
            hasMoved = (State != CursorState.VALID || SkipNext != 0);
            return RC.OK;
        }

        // was:sqlite3BtreeCloseCursor
        public RC Close()
        {
            if (Tree != null)
            {
                Tree.sqlite3BtreeEnter();
                Clear();
                if (Prev != null)
                    Prev.Next = Next;
                else
                    Shared.Cursors = Next;
                if (Next != null)
                    Next.Prev = Prev;
                for (var id = 0; id <= PageID; id++)
                    Pages[id].releasePage();
                Shared.unlockBtreeIfUnused();
                Btree.invalidateOverflowCache(this);
                Tree.sqlite3BtreeLeave();
            }
            return RC.OK;
        }

        // was:sqlite3BtreeKey
        public RC GetKey(uint offset, uint size, byte[] b)
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.VALID);
            Debug.Assert(PageID >= 0 && Pages[PageID] != null);
            Debug.Assert(PagesIndexs[PageID] < Pages[PageID].Cells);
            return AccessPayload(offset, size, b, false);
        }

        // was:sqlite3BtreeData
        public RC GetData(uint offset, uint size, byte[] b)
        {
#if !SQLITE_OMIT_INCRBLOB
            if (State == CursorState.INVALID)
                return RC.ABORT;
#endif
            Debug.Assert(HoldsMutex());
            var rc = RestorePosition();
            if (rc == RC.OK)
            {
                Debug.Assert(State == CursorState.VALID);
                Debug.Assert(PageID >= 0 && Pages[PageID] != null);
                Debug.Assert(PagesIndexs[PageID] < Pages[PageID].Cells);
                rc = AccessPayload(offset, size, b, false);
            }
            return rc;
        }

        // was:sqlite3BtreeKeyFetch
        public byte[] FetchKey(ref int size, out int offset)
        {
            Debug.Assert(MutexEx.Held(Tree.DB.Mutex));
            Debug.Assert(HoldsMutex());
            if (Check.ALWAYS(State == CursorState.VALID))
                return FetchPayload(ref size, out offset, false);
            offset = 0;
            return null;
        }

        // was:sqlite3BtreeDataFetch
        public byte[] FetchData(ref int size, out int offset)
        {
            Debug.Assert(MutexEx.Held(Tree.DB.Mutex));
            Debug.Assert(HoldsMutex());
            if (Check.ALWAYS(State == CursorState.VALID))
                return FetchPayload(ref size, out offset, true);
            offset = 0;
            return null;
        }

        // was:sqlite3BtreeFirst
        public RC MoveFirst(ref bool pRes)
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(MutexEx.Held(Tree.DB.Mutex));
            var rc = MoveToRoot();
            if (rc == RC.OK)
                if (State == CursorState.INVALID)
                {
                    Debug.Assert(Pages[PageID].Cells == 0);
                    pRes = true;
                }
                else
                {
                    Debug.Assert(Pages[PageID].Cells > 0);
                    pRes = false;
                    rc = MoveToLeftmost();
                }
            return rc;
        }

        // was:sqlite3BtreeLast
        public RC MoveLast(ref bool pRes)
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(MutexEx.Held(this.Tree.DB.Mutex));
            // If the cursor already points to the last entry, this is a no-op.
            if (State == CursorState.VALID && AtLast)
            {
#if DEBUG
                // This block serves to Debug.Assert() that the cursor really does point to the last entry in the b-tree.
                for (var id = 0; id < PageID; id++)
                    Debug.Assert(PagesIndexs[id] == Pages[id].Cells);
                Debug.Assert(PagesIndexs[PageID] == Pages[PageID].Cells - 1);
                Debug.Assert(Pages[PageID].Leaf != 0);
#endif
                return RC.OK;
            }
            var rc = MoveToRoot();
            if (rc == RC.OK)
                if (State == CursorState.INVALID)
                {
                    Debug.Assert(Pages[PageID].Cells == 0);
                    pRes = true;
                }
                else
                {
                    Debug.Assert(State == CursorState.VALID);
                    pRes = false;
                    rc = MoveToRightmost();
                    AtLast = (rc == RC.OK);
                }
            return rc;
        }

        // was:sqlite3BtreeMovetoUnpacked
        public RC MoveToUnpacked(Btree.UnpackedRecord idxKey, long intKey, int biasRight, ref int pRes)
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(MutexEx.Held(Tree.DB.Mutex));
            Debug.Assert((idxKey == null) == (KeyInfo == null));
            // If the cursor is already positioned at the point we are trying to move to, then just return without doing any work
            if (State == CursorState.VALID && ValidNKey && Pages[0].HasIntKey)
            {
                if (Info.nKey == intKey)
                {
                    pRes = 0;
                    return RC.OK;
                }
                if (AtLast && Info.nKey < intKey)
                {
                    pRes = -1;
                    return RC.OK;
                }
            }
            var rc = MoveToRoot();
            if (rc != RC.OK)
                return rc;
            Debug.Assert(Pages[PageID] != null);
            Debug.Assert(Pages[PageID].HasInit);
            Debug.Assert(Pages[PageID].Cells > 0 || State == CursorState.INVALID);
            if (State == CursorState.INVALID)
            {
                pRes = -1;
                Debug.Assert(Pages[PageID].Cells == 0);
                return RC.OK;
            }
            Debug.Assert(Pages[0].HasIntKey || idxKey != null);
            for (; ; )
            {

                var page = Pages[PageID];
                // pPage.nCell must be greater than zero. If this is the root-page the cursor would have been INVALID above and this for(;;) loop
                // not run. If this is not the root-page, then the moveToChild() routine would have already detected db corruption. Similarly, pPage must
                // be the right kind (index or table) of b-tree page. Otherwise a moveToChild() or moveToRoot() call would have detected corruption.
                Debug.Assert(page.Cells > 0);
                Debug.Assert(page.HasIntKey == (idxKey == null));
                var lwr = 0;
                var upr = page.Cells - 1;
                int idx;
                PagesIndexs[PageID] = (ushort)(biasRight != 0 ? (idx = upr) : (idx = (upr + lwr) / 2));
                int c;
                for (; ; )
                {
                    Debug.Assert(idx == PagesIndexs[PageID]);
                    Info.nSize = 0;
                    var cell = page.FindCell(idx) + page.ChildPtrSize; // Pointer to current cell in pPage
                    if (page.HasIntKey)
                    {
                        var nCellKey = 0L;
                        if (page.HasData != 0)
                        {
                            uint dummy0;
                            cell += ConvertEx.GetVarint4(page.Data, (uint)cell, out dummy0);
                        }
                        ConvertEx.GetVarint9L(page.Data, (uint)cell, out nCellKey);
                        if (nCellKey == intKey)
                            c = 0;
                        else if (nCellKey < intKey)
                            c = -1;
                        else
                        {
                            Debug.Assert(nCellKey > intKey);
                            c = 1;
                        }
                        ValidNKey = true;
                        Info.nKey = nCellKey;
                    }
                    else
                    {
                        // The maximum supported page-size is 65536 bytes. This means that the maximum number of record bytes stored on an index B-Tree
                        // page is less than 16384 bytes and may be stored as a 2-byte varint. This information is used to attempt to avoid parsing
                        // the entire cell by checking for the cases where the record is stored entirely within the b-tree page by inspecting the first
                        // 2 bytes of the cell.
                        var nCell = (int)page.Data[cell + 0];
                        if (0 == (nCell & 0x80) && nCell <= page.MaxLocal)
                            // This branch runs if the record-size field of the cell is a single byte varint and the record fits entirely on the main b-tree page.
                            c = Btree._vdbe.sqlite3VdbeRecordCompare(nCell, page.Data, cell + 1, idxKey);
                        else if (0 == (page.Data[cell + 1] & 0x80) && (nCell = ((nCell & 0x7f) << 7) + page.Data[cell + 1]) <= page.MaxLocal)
                            // The record-size field is a 2 byte varint and the record fits entirely on the main b-tree page.
                            c = Btree._vdbe.sqlite3VdbeRecordCompare(nCell, page.Data, cell + 2, idxKey);
                        else
                        {
                            // The record flows over onto one or more overflow pages. In this case the whole cell needs to be parsed, a buffer allocated
                            // and accessPayload() used to retrieve the record into the buffer before VdbeRecordCompare() can be called.
                            var pCellBody = new byte[page.Data.Length - cell + page.ChildPtrSize];
                            Buffer.BlockCopy(page.Data, cell - page.ChildPtrSize, pCellBody, 0, pCellBody.Length);
                            page.btreeParseCellPtr(pCellBody, ref Info);
                            nCell = (int)Info.nKey;
                            var pCellKey = MallocEx.sqlite3Malloc(nCell);
                            rc = AccessPayload(0, (uint)nCell, pCellKey, false);
                            if (rc != RC.OK)
                            {
                                pCellKey = null;
                                goto moveto_finish;
                            }
                            c = Btree._vdbe.sqlite3VdbeRecordCompare(nCell, pCellKey, idxKey);
                            pCellKey = null;
                        }
                    }
                    if (c == 0)
                    {
                        if (page.HasIntKey && 0 == page.Leaf)
                        {
                            lwr = idx;
                            upr = lwr - 1;
                            break;
                        }
                        else
                        {
                            pRes = 0;
                            rc = RC.OK;
                            goto moveto_finish;
                        }
                    }
                    if (c < 0)
                        lwr = idx + 1;
                    else
                        upr = idx - 1;
                    if (lwr > upr)
                        break;
                    PagesIndexs[PageID] = (ushort)(idx = (lwr + upr) / 2);
                }
                Debug.Assert(lwr == upr + 1);
                Debug.Assert(page.HasInit);
                Pgno chldPg;
                if (page.Leaf != 0)
                    chldPg = 0;
                else if (lwr >= page.Cells)
                    chldPg = ConvertEx.Get4(page.Data, page.HeaderOffset + 8);
                else
                    chldPg = ConvertEx.Get4(page.Data, page.FindCell(lwr));
                if (chldPg == 0)
                {
                    Debug.Assert(PagesIndexs[PageID] < Pages[PageID].Cells);
                    pRes = c;
                    rc = RC.OK;
                    goto moveto_finish;
                }
                PagesIndexs[PageID] = (ushort)lwr;
                Info.nSize = 0;
                ValidNKey = false;
                rc = MoveToChild(chldPg);
                if (rc != RC.OK)
                    goto moveto_finish;
            }
        moveto_finish:
            return rc;
        }

        // was:sqlite3BtreeEof
        public bool Eof
        {
            // TODO: What if the cursor is in CURSOR_REQUIRESEEK but all table entries have been deleted? This API will need to change to return an error code
            // as well as the boolean result value.
            get { return (State != CursorState.VALID); }
        }

        // was:sqlite3BtreeNext
        public RC MoveNext(ref int pRes)
        {
            Debug.Assert(HoldsMutex());
            var rc = RestorePosition();
            if (rc != RC.OK)
                return rc;
            if (State == CursorState.INVALID)
            {
                pRes = 1;
                return RC.OK;
            }
            if (SkipNext > 0)
            {
                SkipNext = 0;
                pRes = 0;
                return RC.OK;
            }
            SkipNext = 0;
            //
            var page = Pages[PageID];
            var index = ++PagesIndexs[PageID];
            Debug.Assert(page.HasInit);
            Debug.Assert(index <= page.Cells);
            Info.nSize = 0;
            ValidNKey = false;
            if (index >= page.Cells)
            {
                if (page.Leaf == 0)
                {
                    rc = MoveToChild(ConvertEx.Get4(page.Data, page.HeaderOffset + 8));
                    if (rc != RC.OK)
                        return rc;
                    rc = MoveToLeftmost();
                    pRes = 0;
                    return rc;
                }
                do
                {
                    if (PageID == 0)
                    {
                        pRes = 1;
                        State = CursorState.INVALID;
                        return RC.OK;
                    }
                    MoveToParent();
                    page = Pages[PageID];
                } while (PagesIndexs[PageID] >= page.Cells);
                pRes = 0;
                rc = (page.HasIntKey ? MoveNext(ref pRes) : RC.OK);
                return rc;
            }
            pRes = 0;
            if (page.Leaf != 0)
                return RC.OK;
            rc = MoveToLeftmost();
            return rc;
        }

        // was:sqlite3BtreePrevious
        public RC MovePrevious(ref int pRes)
        {
            Debug.Assert(HoldsMutex());
            var rc = RestorePosition();
            if (rc != RC.OK)
                return rc;
            AtLast = false;
            if (State == CursorState.INVALID)
            {
                pRes = 1;
                return RC.OK;
            }
            if (SkipNext < 0)
            {
                SkipNext = 0;
                pRes = 0;
                return RC.OK;
            }
            SkipNext = 0;
            //
            var page = Pages[PageID];
            Debug.Assert(page.HasInit);
            if (page.Leaf == 0)
            {
                var index = PagesIndexs[PageID];
                rc = MoveToChild(ConvertEx.Get4(page.Data, page.FindCell(index)));
                if (rc != RC.OK)
                    return rc;
                rc = MoveToRightmost();
            }
            else
            {
                while (PagesIndexs[PageID] == 0)
                {
                    if (PageID == 0)
                    {
                        State = CursorState.INVALID;
                        pRes = 1;
                        return RC.OK;
                    }
                    MoveToParent();
                }
                Info.nSize = 0;
                ValidNKey = false;
                PagesIndexs[PageID]--;
                page = Pages[PageID];
                rc = (page.HasIntKey && page.Leaf == 0 ? MovePrevious(ref pRes) : RC.OK);
            }
            pRes = 0;
            return rc;
        }

#if !SQLITE_OMIT_BTREECOUNT
        // was:sqlite3BtreeCount
        public RC GetCount(ref long count)
        {
            long count2 = 0; // Value to return in pnEntry
            var rc = MoveToRoot();
            // Unless an error occurs, the following loop runs one iteration for each page in the B-Tree structure (not including overflow pages).
            while (rc == RC.OK)
            {
                // If this is a leaf page or the tree is not an int-key tree, then this page contains countable entries. Increment the entry counter accordingly.
                var page = Pages[PageID]; // Current page of the b-tree 
                if (page.Leaf != 0 || !page.HasIntKey)
                    count2 += page.Cells;
                // pPage is a leaf node. This loop navigates the cursor so that it points to the first interior cell that it points to the parent of
                // the next page in the tree that has not yet been visited. The pCur.aiIdx[pCur.iPage] value is set to the index of the parent cell
                // of the page, or to the number of cells in the page if the next page to visit is the right-child of its parent.
                // If all pages in the tree have been visited, return SQLITE_OK to the caller.
                if (page.Leaf != 0)
                {
                    do
                    {
                        if (PageID == 0)
                        {
                            // All pages of the b-tree have been visited. Return successfully.
                            count = count2;
                            return RC.OK;
                        }
                        MoveToParent();
                    } while (PagesIndexs[PageID] >= Pages[PageID].Cells);
                    PagesIndexs[PageID]++;
                    page = Pages[PageID];
                }
                // Descend to the child node of the cell that the cursor currently points at. This is the right-child if (iIdx==pPage.nCell).
                var indexID = PagesIndexs[PageID]; // Index of child node in parent
                rc = (indexID == page.Cells ? MoveToChild(ConvertEx.Get4(page.Data, page.HeaderOffset + 8)) : MoveToChild(ConvertEx.Get4(page.Data, page.FindCell(indexID))));
            }
            // An error has occurred. Return an error code.
            return rc;
        }
#endif
    }
}
