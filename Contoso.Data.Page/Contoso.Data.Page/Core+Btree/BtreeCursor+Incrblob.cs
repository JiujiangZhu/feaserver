using System.Diagnostics;
using Contoso.Sys;
using TRANS = Contoso.Core.Btree.TRANS;
using LOCK = Contoso.Core.Btree.BtreeLock.LOCK;

namespace Contoso.Core
{
    public partial class BtreeCursor
    {
#if !SQLITE_OMIT_INCRBLOB
        // was:sqlite3BtreePutData
        public RC BtreePutData(uint offset, uint amt, byte[] z)
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(MutexEx.Held(Tree.DB.Mutex));
            Debug.Assert(IsIncrblob);
            var rc = RestorePosition();
            if (rc != RC.OK)
                return rc;
            Debug.Assert(State != CursorState.REQUIRESEEK);
            if (State != CursorState.VALID)
                return RC.ABORT;
            // Check some assumptions:
            //   (a) the cursor is open for writing,
            //   (b) there is a read/write transaction open,
            //   (c) the connection holds a write-lock on the table (if required),
            //   (d) there are no conflicting read-locks, and
            //   (e) the cursor points at a valid row of an intKey table.
            if (!Writeable)
                return RC.READONLY;
            Debug.Assert(!Shared.ReadOnly && Shared.InTransaction == TRANS.WRITE);
            Debug.Assert(Tree.hasSharedCacheTableLock(RootID, false, LOCK.WRITE));
            Debug.Assert(!Tree.hasReadConflicts(RootID));
            Debug.Assert(Pages[PageID].HasIntKey);
            return AccessPayload(offset, amt, z, true);
        }

        // was:sqlite3BtreeCacheOverflow
        public void BtreeCacheOverflow()
        {
            Debug.Assert(HoldsMutex());
            Debug.Assert(MutexEx.Held(Tree.DB.Mutex));
            Btree.invalidateOverflowCache(this);
            IsIncrblob = true;
        }
#endif
    }
}
