using System.Diagnostics;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;

namespace Contoso.Core
{
    public partial class Btree
    {
#if !SQLITE_OMIT_INCRBLOB
        // was:invalidateOverflowCache
        internal static void invalidateOverflowCache(BtreeCursor cursor)
        {
            Debug.Assert(cursor.HoldsMutex());
            cursor.OverflowIDs = null;
        }

        // was:invalidateAllOverflowCache
        internal static void invalidateAllOverflowCache(BtShared shared)
        {
            Debug.Assert(MutexEx.Held(shared.Mutex));
            for (var p = shared.Cursors; p != null; p = p.Next)
                invalidateOverflowCache(p);
        }

        // was:invalidateIncrblobCursors
        internal static void invalidateIncrblobCursors(Btree tree, long row, bool clearTable)
        {
            var pBt = tree.Shared;
            Debug.Assert(tree.sqlite3BtreeHoldsMutex());
            for (var p = pBt.Cursors; p != null; p = p.Next)
                if (p.IsIncrblob && (clearTable || p.Info.nKey == row))
                    p.State = CURSOR.INVALID;
        }
#else
        // was:invalidateOverflowCache
        internal static void invalidateOverflowCache(BtCursor pCur) { }
        // was:invalidateAllOverflowCache
        internal static void invalidateAllOverflowCache(BtShared pBt) { }
        // was:invalidateIncrblobCursors
        internal static void invalidateIncrblobCursors(Btree tree, long row, int clearTable) { }
#endif
    }
}
