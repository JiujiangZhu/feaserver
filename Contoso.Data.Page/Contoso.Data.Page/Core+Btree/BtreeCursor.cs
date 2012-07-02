using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;

namespace Contoso.Core
{
    // was:BtCursor
    public partial class BtreeCursor
    {
        private const int BTCURSOR_MAX_DEPTH = 20;

        public enum CursorState : int
        {
            INVALID = 0,
            VALID = 1,
            REQUIRESEEK = 2,
            FAULT = 3,
        }

        public Btree Tree;                  // The Btree to which this cursor belongs
        public BtShared Shared;             // The BtShared this cursor points to
        public BtreeCursor Next;
        public BtreeCursor Prev;
        public KeyInfo KeyInfo;             // Argument passed to comparison function
        public Pgno RootID;                 // The root page of this tree
        public long _cachedRowID;            // Next rowid cache.  0 means not valid
        public MemPage.CellInfo Info = new MemPage.CellInfo();      // A parse of the cell we are pointing at
        public byte[] Key;                 // Saved key that was cursor's last known position
        public long NKey;                   // Size of pKey, or last integer key
        public RC SkipNext;                // Prev() is noop if negative. Next() is noop if positive
        public bool Writeable;              // True if writable
        public bool AtLast;                 // VdbeCursor pointing to the last entry
        public bool ValidNKey;              // True if info.nKey is valid
        public CursorState State;           // One of the CURSOR_XXX constants (see below)
#if !SQLITE_OMIT_INCRBLOB
        public uint[] OverflowIDs;          // Cache of overflow page locations
        public bool IsIncrblob;       // True if this cursor is an incr. io handle
#endif
        public short PageID;                                            // Index of current page in apPage
        public ushort[] PagesIndexs = new ushort[BTCURSOR_MAX_DEPTH];   // Current index in apPage[i]
        public MemPage[] Pages = new MemPage[BTCURSOR_MAX_DEPTH];       // Pages from root to current page

        // was:sqlite3BtreeCursorZero
        public void Zero()
        {
            Next = null;
            Prev = null;
            KeyInfo = null;
            RootID = 0;
            _cachedRowID = 0;
            Info = new MemPage.CellInfo();
            Writeable = false;
            AtLast = false;
            ValidNKey = false;
            State = 0;
            Key = null;
            NKey = 0;
            SkipNext = 0;
#if !SQLITE_OMIT_INCRBLOB
            IsIncrblob = false;
            OverflowIDs = null;
#endif
            PageID = 0;
        }

        private BtreeCursor Clone() { return (BtreeCursor)MemberwiseClone(); }

#if DEBUG
        // was:cursorHoldsMutex
        internal bool HoldsMutex() { return MutexEx.Held(this.Shared.Mutex); }
#else
        // was:cursorHoldsMutex
        internal bool HoldsMutex() { return true; }
#endif


    }
}
