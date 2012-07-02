using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class BtreeCursor
    {
        // was:saveCursorPosition
        internal RC SavePosition()
        {
            Debug.Assert(State == CursorState.VALID);
            Debug.Assert(Key == null);
            Debug.Assert(HoldsMutex());
            NKey = GetKeySize();
            // If this is an intKey table, then the above call to BtreeKeySize() stores the integer key in pCur.nKey. In this case this value is
            // all that is required. Otherwise, if pCur is not open on an intKey table, then malloc space for and store the pCur.nKey bytes of key data.
            var rc = RC.OK;
            if (!Pages[0].HasIntKey)
            {
                var pKey = MallocEx.sqlite3Malloc((int)NKey);
                rc = GetKey(0, (uint)NKey, pKey);
                if (rc == RC.OK)
                    Key = pKey;
            }
            Debug.Assert(!Pages[0].HasIntKey || Key == null);
            if (rc == RC.OK)
            {
                for (var i = 0; i <= PageID; i++)
                {
                    Pages[i].releasePage();
                    Pages[i] = null;
                }
                PageID = -1;
                State = CursorState.REQUIRESEEK;
            }
            Btree.invalidateOverflowCache(this);
            return rc;
        }

        // was:btreeMoveto
        private RC BtreeMoveTo(byte[] key, long nKey, bool biasRight, ref int pRes)
        {
            Btree.UnpackedRecord pIdxKey; // Unpacked index key
            var aSpace = new Btree.UnpackedRecord();
            if (key != null)
            {
                Debug.Assert(nKey == (long)(int)nKey);
                pIdxKey = Btree._vdbe.sqlite3VdbeRecordUnpack(this.KeyInfo, (int)nKey, key, aSpace, 16);
            }
            else
                pIdxKey = null;
            var rc = MoveToUnpacked(pIdxKey, nKey, biasRight, ref pRes);
            if (key != null)
                Btree._vdbe.sqlite3VdbeDeleteUnpackedRecord(pIdxKey);
            return rc;
        }

        // was:btreeRestoreCursorPosition
        private RC BtreeRestorePosition()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State >= CursorState.REQUIRESEEK);
            if (State == CursorState.FAULT)
                return SkipNext;
            State = CursorState.INVALID;
            RC rc; { int skipNextAsInt = (int)SkipNext; rc = BtreeMoveTo(Key, NKey, false, ref skipNextAsInt); SkipNext = (RC)skipNextAsInt; }
            if (rc == RC.OK)
            {
                Key = null;
                Debug.Assert(State == CursorState.VALID || State == CursorState.INVALID);
            }
            return rc;
        }

        // was:restoreCursorPosition
        private RC RestorePosition() { return (State >= CursorState.REQUIRESEEK ? BtreeRestorePosition() : RC.OK); }

        // was:moveToChild
        private RC MoveToChild(Pgno newID)
        {
            var id = PageID;
            var newPage = new MemPage();
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.VALID);
            Debug.Assert(PageID < BTCURSOR_MAX_DEPTH);
            if (PageID >= (BTCURSOR_MAX_DEPTH - 1))
                return SysEx.SQLITE_CORRUPT_BKPT();
            var rc = Shared.getAndInitPage(newID, ref newPage);
            if (rc != RC.OK)
                return rc;
            Pages[id + 1] = newPage;
            PagesIndexs[id + 1] = 0;
            PageID++;
            Info.nSize = 0;
            ValidNKey = false;
            if (newPage.Cells < 1 || newPage.HasIntKey != Pages[id].HasIntKey)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return RC.OK;
        }

        // was:moveToParent
        private void MoveToParent()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.VALID);
            Debug.Assert(PageID > 0);
            Debug.Assert(Pages[PageID] != null);
            MemPage.assertParentIndex(Pages[PageID - 1], PagesIndexs[PageID - 1], Pages[PageID].ID);
            Pages[PageID].releasePage();
            PageID--;
            Info.nSize = 0;
            ValidNKey = false;
        }

        // was:moveToRoot
        private RC MoveToRoot()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(CursorState.INVALID < CursorState.REQUIRESEEK);
            Debug.Assert(CursorState.VALID < CursorState.REQUIRESEEK);
            Debug.Assert(CursorState.FAULT > CursorState.REQUIRESEEK);
            if (State >= CursorState.REQUIRESEEK)
            {
                if (State == CursorState.FAULT)
                {
                    Debug.Assert(SkipNext != 0);
                    return (RC)SkipNext;
                }
                Clear();
            }
            var rc = RC.OK;
            if (PageID >= 0)
            {
                for (var id = 1; id <= PageID; id++)
                    Pages[id].releasePage();
                PageID = 0;
            }
            else
            {
                rc = Tree.Shared.getAndInitPage(RootID, ref Pages[0]);
                if (rc != RC.OK)
                {
                    State = CursorState.INVALID;
                    return rc;
                }
                PageID = 0;
                // If pCur.pKeyInfo is not NULL, then the caller that opened this cursor expected to open it on an index b-tree. Otherwise, if pKeyInfo is
                // NULL, the caller expects a table b-tree. If this is not the case, return an SQLITE_CORRUPT error.
                if ((KeyInfo == null) != (Pages[0].HasIntKey))
                    return SysEx.SQLITE_CORRUPT_BKPT();
            }
            // Assert that the root page is of the correct type. This must be the case as the call to this function that loaded the root-page (either
            // this call or a previous invocation) would have detected corruption if the assumption were not true, and it is not possible for the flags
            // byte to have been modified while this cursor is holding a reference to the page.
            var root = Pages[0];
            Debug.Assert(root.ID == RootID);
            Debug.Assert(root.HasInit && (KeyInfo == null) == root.HasIntKey);
            PagesIndexs[0] = 0;
            Info.nSize = 0;
            AtLast = false;
            ValidNKey = false;
            if (root.Cells == 0 && 0 == root.Leaf)
            {
                if (root.ID != 1)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                var subpage = (Pgno)ConvertEx.Get4(root.Data, root.HeaderOffset + 8);
                State = CursorState.VALID;
                rc = MoveToChild(subpage);
            }
            else
                State = (root.Cells > 0 ? CursorState.VALID : CursorState.INVALID);
            return rc;
        }

        // was:moveToLeftmost
        private RC MoveToLeftmost()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.VALID);
            var rc = RC.OK;
            MemPage page;
            while (rc == RC.OK && (page = Pages[PageID]).Leaf == 0)
            {
                Debug.Assert(PagesIndexs[PageID] < page.Cells);
                var id = (Pgno)ConvertEx.Get4(page.Data, page.FindCell(PagesIndexs[PageID]));
                rc = MoveToChild(id);
            }
            return rc;
        }

        // was:moveToRightmost
        private RC MoveToRightmost()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.VALID);
            var rc = RC.OK;
            MemPage page = null;
            while (rc == RC.OK && (page = Pages[PageID]).Leaf == 0)
            {
                var pgno = (Pgno)ConvertEx.Get4(page.Data, page.HeaderOffset + 8);
                PagesIndexs[PageID] = page.Cells;
                rc = MoveToChild(pgno);
            }
            if (rc == RC.OK)
            {
                PagesIndexs[PageID] = (ushort)(page.Cells - 1);
                Info.nSize = 0;
                ValidNKey = false;
            }
            return rc;
        }
    }
}
