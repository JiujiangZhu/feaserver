using Contoso.Sys;
using Pgno = System.UInt32;
using LOCK = Contoso.Core.Btree.BtreeLock.LOCK;
using System.Diagnostics;
using System;

namespace Contoso.Core
{
    public partial class Btree
    {
#if !SQLITE_OMIT_SHARED_CACHE
        internal static BtShared s_sqlite3SharedCacheList = null;

        private static bool g_sharedCacheEnabled;

        internal RC sqlite3_enable_shared_cache(bool enable) { g_sharedCacheEnabled = enable; return RC.OK; }

#if DEBUG
        internal bool hasSharedCacheTableLock(Pgno rootID, bool isIndex, LOCK eLockType)
        {
            // If this database is not shareable, or if the client is reading and has the read-uncommitted flag set, then no lock is required.  Return true immediately.
            if (!Sharable || (eLockType == LOCK.READ && (DB.flags & sqlite3.SQLITE.ReadUncommitted) != 0))
                return true;
            // If the client is reading or writing an index and the schema is not loaded, then it is too difficult to actually check to see if
            // the correct locks are held.  So do not bother - just return true. This case does not come up very often anyhow.
            var schema = Shared.Schema;
            if (isIndex && (schema == null || (schema.flags & SchemaFlags.SchemaLoaded) == 0))
                return true;
            // Figure out the root-page that the lock should be held on. For table b-trees, this is just the root page of the b-tree being read or
            // written. For index b-trees, it is the root page of the associated table.
            Pgno tableID = 0;
            if (isIndex)
                throw new NotImplementedException();
                //for (var p = schema.idxHash.first; p != null; p = p.next)
                //{
                //    var pIdx = (IIndex)p.data;
                //    if (pIdx.tnum == (int)rootID)
                //        tableID = pIdx.pTable.tnum;
                //}
            else
                tableID = rootID;
            // Search for the required lock. Either a write-lock on root-page iTab, a write-lock on the schema table, or (if the client is reading) a
            // read-lock on tableID will suffice. Return 1 if any of these are found.
            for (var @lock = Shared.Locks; @lock != null; @lock = @lock.Next)
                if (@lock.Tree == this && (@lock.TableID == tableID || (@lock.Lock == LOCK.WRITE && @lock.TableID == 1)) && @lock.Lock >= eLockType)
                    return true;
            // Failed to find the required lock.
            return false;
        }

        internal bool hasReadConflicts(Pgno rootID)
        {
            for (var cursor = Shared.Cursors; cursor != null; cursor = cursor.Next)
                if (cursor.RootID == rootID && cursor.Tree != this && (cursor.Tree.DB.flags & sqlite3.SQLITE.ReadUncommitted) == 0)
                    return true;
            return false;
        }
#endif

        internal RC querySharedCacheTableLock(Pgno tableID, LOCK eLock)
        {
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(eLock == LOCK.READ || eLock == LOCK.WRITE);
            Debug.Assert(DB != null);
            Debug.Assert((DB.flags & sqlite3.SQLITE.ReadUncommitted) == 0 || eLock == LOCK.WRITE || tableID == 1);
            // If requesting a write-lock, then the Btree must have an open write transaction on this file. And, obviously, for this to be so there
            // must be an open write transaction on the file itself.
            var shared = this.Shared;
            Debug.Assert(eLock == LOCK.READ || (shared.Writer == this && InTransaction == TRANS.WRITE));
            Debug.Assert(eLock == LOCK.READ || shared.InTransaction == TRANS.WRITE);
            // This routine is a no-op if the shared-cache is not enabled 
            if (!Sharable)
                return RC.OK;
            // If some other connection is holding an exclusive lock, the requested lock may not be obtained.
            if (shared.Writer != this && shared.IsExclusive)
            {
                sqlite3.sqlite3ConnectionBlocked(DB, shared.Writer.DB);
                return RC.LOCKED_SHAREDCACHE;
            }
            for (var @lock = shared.Locks; @lock != null; @lock = @lock.Next)
            {
                // The condition (pIter.eLock!=eLock) in the following if(...) statement is a simplification of:
                //   (eLock==WRITE_LOCK || pIter.eLock==WRITE_LOCK)
                // since we know that if eLock==WRITE_LOCK, then no other connection may hold a WRITE_LOCK on any table in this file (since there can
                // only be a single writer).
                Debug.Assert(@lock.Lock == LOCK.READ || @lock.Lock == LOCK.WRITE);
                Debug.Assert(eLock == LOCK.READ || @lock.Tree == this || @lock.Lock == LOCK.READ);
                if (@lock.Tree != this && @lock.TableID == tableID && @lock.Lock != eLock)
                {
                    sqlite3.sqlite3ConnectionBlocked(DB, @lock.Tree.DB);
                    if (eLock == LOCK.WRITE)
                    {
                        Debug.Assert(shared.Writer == this);
                        shared.IsPending = true;
                    }
                    return RC.LOCKED_SHAREDCACHE;
                }
            }
            return RC.OK;
        }

        internal RC setSharedCacheTableLock(Pgno tableID, LOCK eLock)
        {
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(eLock == LOCK.READ || eLock == LOCK.WRITE);
            Debug.Assert(DB != null);
            // A connection with the read-uncommitted flag set will never try to obtain a read-lock using this function. The only read-lock obtained
            // by a connection in read-uncommitted mode is on the sqlite_master table, and that lock is obtained in BtreeBeginTrans().
            Debug.Assert((DB.flags & sqlite3.SQLITE.ReadUncommitted) == 0 || eLock == LOCK.WRITE);
            // This function should only be called on a sharable b-tree after it has been determined that no other b-tree holds a conflicting lock.
            Debug.Assert(Sharable);
            Debug.Assert(querySharedCacheTableLock(tableID, eLock) == RC.OK);
            // First search the list for an existing lock on this table.
            BtreeLock pLock = null;
            for (var @lock = Shared.Locks; @lock != null; @lock = @lock.Next)
                if (@lock.TableID == tableID && @lock.Tree == this)
                {
                    pLock = @lock;
                    break;
                }
            // If the above search did not find a BtLock struct associating Btree p with table iTable, allocate one and link it into the list.
            if (pLock == null)
            {
                var shared = Shared;
                pLock = new BtreeLock();
                pLock.TableID = tableID;
                pLock.Tree = this;
                pLock.Next = shared.Locks;
                shared.Locks = pLock;
            }
            // Set the BtLock.eLock variable to the maximum of the current lock and the requested lock. This means if a write-lock was already held
            // and a read-lock requested, we don't incorrectly downgrade the lock.
            Debug.Assert(LOCK.WRITE > LOCK.READ);
            if (eLock > pLock.Lock)
                pLock.Lock = eLock;
            return RC.OK;
        }

        internal void clearAllSharedCacheTableLocks()
        {
            var shared = Shared;
            var ppIter = shared.Locks;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(Sharable || ppIter == null);
            Debug.Assert(InTransaction > 0);
            while (ppIter != null)
            {
                var pLock = ppIter;
                Debug.Assert(shared.IsExclusive == false || shared.Writer == pLock.Tree);
                Debug.Assert((int)pLock.Tree.InTransaction >= (int)pLock.Lock);
                if (pLock.Tree == this)
                {
                    ppIter = pLock.Next;
                    Debug.Assert(pLock.TableID != 1 || pLock == this.Locks);
                    if (pLock.TableID != 1)
                        pLock = null;//sqlite3_free(ref pLock);
                }
                else
                    ppIter = pLock.Next;
            }
            Debug.Assert(shared.IsPending == null || shared.Writer != null);
            if (shared.Writer == this)
            {
                shared.Writer = null;
                shared.IsExclusive = false;
                shared.IsPending = false;
            }
            else if (shared.Transactions == 2)
                // This function is called when Btree p is concluding its transaction. If there currently exists a writer, and p is not
                // that writer, then the number of locks held by connections other than the writer must be about to drop to zero. In this case
                // set the isPending flag to 0.
                // If there is not currently a writer, then BtShared.isPending must be zero already. So this next line is harmless in that case.
                shared.IsPending = false;
        }

        internal void downgradeAllSharedCacheTableLocks()
        {
            var shared = Shared;
            if (shared.Writer == this)
            {
                shared.Writer = null;
                shared.IsExclusive = false;
                shared.IsPending = false;
                for (var @lock = shared.Locks; @lock != null; @lock = @lock.Next)
                {
                    Debug.Assert(@lock.Lock == LOCK.READ || @lock.Tree == this);
                    @lock.Lock = LOCK.READ;
                }
            }
        }
#else
        internal RC querySharedCacheTableLock(Pgno iTab, BtreeLock.LOCK eLock) { return RC.OK; }
        internal void clearAllSharedCacheTableLocks() { }
        internal void downgradeAllSharedCacheTableLocks() { }
        internal bool hasSharedCacheTableLock(Pgno b, bool c, LOCK d) { return true; }
        internal bool hasReadConflicts(Pgno b) { return false; }
#endif
    }
}
