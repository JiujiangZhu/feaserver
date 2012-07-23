using System.Diagnostics;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using Contoso.Threading;

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
            for (var cursor = shared.Cursors; cursor != null; cursor = cursor.Next)
                invalidateOverflowCache(cursor);
        }

        // was:invalidateIncrblobCursors
        internal static void invalidateIncrblobCursors(Btree tree, long row, bool clearTable)
        {
            var shared = tree.Shared;
            Debug.Assert(tree.sqlite3BtreeHoldsMutex());
            for (var cursor = shared.Cursors; cursor != null; cursor = cursor.Next)
                if (cursor.IsIncrblob && (clearTable || cursor.Info.nKey == row))
                    cursor.State = CURSOR.INVALID;
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
