using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class BtreeCursor
    {
        // was:sqlite3BtreeCursorSize
        public int sqlite3BtreeCursorSize() { return -1; }

        // was:sqlite3BtreeGetCachedRowid/sqlite3BtreeSetCachedRowid
        public long CachedRowID
        {
            get { return _cachedRowID; }
            set
            {
                for (var p = Shared.Cursors; p != null; p = p.Next)
                    if (p.RootID == RootID)
                        p._cachedRowID = value;
                Debug.Assert(_cachedRowID == value);
            }
        }

#if DEBUG
        // was:sqlite3BtreeCursorIsValid
        public bool IsValid { get { return this != null && this.State == CursorState.VALID; } }
#else
        // was:sqlite3BtreeCursorIsValid
        public bool IsValid { get { return true; } }
#endif

        // was:sqlite3BtreeKeySize
        public long GetKeySize()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.INVALID || State == CursorState.VALID);
            if (State != CursorState.VALID)
                return 0;
            GetCellInfo();
            return Info.nKey;
        }

        // was:sqlite3BtreeDataSize
        public uint GetDataSize()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(State == CursorState.VALID);
            GetCellInfo();
            return Info.nData;
        }
    }
}
