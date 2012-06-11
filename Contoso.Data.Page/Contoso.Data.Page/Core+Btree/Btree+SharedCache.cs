using Contoso.Sys;
using Pgno = System.UInt32;
using LOCK = Contoso.Core.Btree.BtreeLock.LOCK;

namespace Contoso.Core
{
    public partial class Btree
    {
#if !SQLITE_OMIT_SHARED_CACHE
        internal RC sqlite3_enable_shared_cache(int enable) { sqlite3GlobalConfig.sharedCacheEnabled = enable; return RC.OK; }
#if DEBUG
        internal int hasSharedCacheTableLock(Pgno iRoot, int isIndex, LOCK eLockType)
        {
            var schema = Shared.Schema;
            Pgno iTab = 0;
            // If this database is not shareable, or if the client is reading and has the read-uncommitted flag set, then no lock is required.  Return true immediately.
            if ((sharable == null) || (eLockType == LOCK.READ && (DB.flags & sqlite3.SQLITE.ReadUncommitted)))
                return 1;
            // If the client is reading  or writing an index and the schema is not loaded, then it is too difficult to actually check to see if
            // the correct locks are held.  So do not bother - just return true. This case does not come up very often anyhow.
            if (isIndex && (schema == null || (schema.flags & DB_SchemaLoaded) == 0))
                return 1;
            // Figure out the root-page that the lock should be held on. For table b-trees, this is just the root page of the b-tree being read or
            // written. For index b-trees, it is the root page of the associated table.
            if (isIndex)
            {
                for (var p = sqliteHashFirst(schema.idxHash); p != null; p = sqliteHashNext(p))
                {
                    var pIdx = (Index)sqliteHashData(p);
                    if (pIdx.tnum == (int)iRoot)
                        iTab = pIdx.pTable.tnum;
                }
            }
            else
                iTab = iRoot;
            // Search for the required lock. Either a write-lock on root-page iTab, a write-lock on the schema table, or (if the client is reading) a
            // read-lock on iTab will suffice. Return 1 if any of these are found.
            for (var pLock = Shared.Locks; pLock != null; pLock = pLock.Next)
                if (pLock.pBtree == this && (pLock.iTable == iTab || (pLock.eLock == LOCK.WRITE && pLock.iTable == 1)) && pLock.eLock >= eLockType)
                    return 1;
            // Failed to find the required lock.
            return 0;
        }

        internal int hasReadConflicts(Pgno rootID)
        {
            for (var p = Shared.Cursors; p != null; p = p.Next)
                if (p.RootID == rootID && p.Tree != this && (p.Tree.DB.flags & sqlite3.SQLITE.ReadUncommitted) == 0)
                    return 1;
            return 0;
        }
#endif

        static int querySharedCacheTableLock(Btree p, Pgno iTab, LOCK eLock)
        {
            BtShared pBt = p.pBt;
            BtLock pIter;
            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            Debug.Assert(eLock == READ_LOCK || eLock == WRITE_LOCK);
            Debug.Assert(p.db != null);
            Debug.Assert(!(p.db.flags & SQLITE_ReadUncommitted) || eLock == WRITE_LOCK || iTab == 1);
            // If requesting a write-lock, then the Btree must have an open write transaction on this file. And, obviously, for this to be so there
            // must be an open write transaction on the file itself.
            Debug.Assert(eLock == READ_LOCK || (p == pBt.pWriter && p.inTrans == TRANS_WRITE));
            Debug.Assert(eLock == READ_LOCK || pBt.inTransaction == TRANS_WRITE);
            // This routine is a no-op if the shared-cache is not enabled 
            if (!p.sharable)
            {
                return SQLITE_OK;
            }
            // If some other connection is holding an exclusive lock, the requested lock may not be obtained.
            if (pBt.pWriter != p && pBt.isExclusive)
            {
                sqlite3ConnectionBlocked(p.db, pBt.pWriter.db);
                return SQLITE_LOCKED_SHAREDCACHE;
            }
            for (pIter = pBt.pLock; pIter; pIter = pIter.pNext)
            {
                // The condition (pIter.eLock!=eLock) in the following if(...) statement is a simplification of:
                //   (eLock==WRITE_LOCK || pIter.eLock==WRITE_LOCK)
                // since we know that if eLock==WRITE_LOCK, then no other connection may hold a WRITE_LOCK on any table in this file (since there can
                // only be a single writer).
                Debug.Assert(pIter.eLock == READ_LOCK || pIter.eLock == WRITE_LOCK);
                Debug.Assert(eLock == READ_LOCK || pIter.pBtree == p || pIter.eLock == READ_LOCK);
                if (pIter.pBtree != p && pIter.iTable == iTab && pIter.eLock != eLock)
                {
                    sqlite3ConnectionBlocked(p.db, pIter.pBtree.db);
                    if (eLock == WRITE_LOCK)
                    {
                        Debug.Assert(p == pBt.pWriter);
                        pBt.isPending = 1;
                    }
                    return SQLITE_LOCKED_SHAREDCACHE;
                }
            }
            return SQLITE_OK;
        }

        static int setSharedCacheTableLock(Btree p, Pgno iTable, LOCK eLock)
        {
            BtShared pBt = p.pBt;
            BtLock pLock = 0;
            BtLock pIter;
            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            Debug.Assert(eLock == READ_LOCK || eLock == WRITE_LOCK);
            Debug.Assert(p.db != null);
            // A connection with the read-uncommitted flag set will never try to obtain a read-lock using this function. The only read-lock obtained
            // by a connection in read-uncommitted mode is on the sqlite_master table, and that lock is obtained in BtreeBeginTrans().
            Debug.Assert(0 == (p.db.flags & SQLITE_ReadUncommitted) || eLock == WRITE_LOCK);
            // This function should only be called on a sharable b-tree after it has been determined that no other b-tree holds a conflicting lock.
            Debug.Assert(p.sharable);
            Debug.Assert(SQLITE_OK == querySharedCacheTableLock(p, iTable, eLock));
            // First search the list for an existing lock on this table.
            for (pIter = pBt.pLock; pIter; pIter = pIter.pNext)
            {
                if (pIter.iTable == iTable && pIter.pBtree == p)
                {
                    pLock = pIter;
                    break;
                }
            }
            // If the above search did not find a BtLock struct associating Btree p with table iTable, allocate one and link it into the list.
            if (!pLock)
            {
                pLock = (BtLock*)sqlite3MallocZero(sizeof(BtLock));
                if (!pLock)
                {
                    return SQLITE_NOMEM;
                }
                pLock.iTable = iTable;
                pLock.pBtree = p;
                pLock.pNext = pBt.pLock;
                pBt.pLock = pLock;
            }
            // Set the BtLock.eLock variable to the maximum of the current lock and the requested lock. This means if a write-lock was already held
            // and a read-lock requested, we don't incorrectly downgrade the lock.
            Debug.Assert(WRITE_LOCK > READ_LOCK);
            if (eLock > pLock.eLock)
            {
                pLock.eLock = eLock;
            }
            return SQLITE_OK;
        }
        static void clearAllSharedCacheTableLocks(Btree p){
BtShared pBt = p.pBt;
BtLock **ppIter = &pBt.pLock;
Debug.Assert( sqlite3BtreeHoldsMutex(p) );
Debug.Assert( p.sharable || 0==*ppIter );
Debug.Assert( p.inTrans>0 );
while( ppIter ){
BtLock pLock = ppIter;
Debug.Assert( pBt.isExclusive==null || pBt.pWriter==pLock.pBtree );
Debug.Assert( pLock.pBtree.inTrans>=pLock.eLock );
if( pLock.pBtree==p ){
ppIter = pLock.pNext;
Debug.Assert( pLock.iTable!=1 || pLock==&p.Locks );
if( pLock.iTable!=1 ){
pLock=null;//sqlite3_free(ref pLock);
}
}else{
ppIter = &pLock.pNext;
}
}

Debug.Assert( pBt.isPending==null || pBt.pWriter );
if( pBt.pWriter==p ){
pBt.pWriter = 0;
pBt.isExclusive = 0;
pBt.isPending = 0;
}else if( pBt.nTransaction==2 ){
/* This function is called when Btree p is concluding its 
** transaction. If there currently exists a writer, and p is not
** that writer, then the number of locks held by connections other
** than the writer must be about to drop to zero. In this case
** set the isPending flag to 0.
**
** If there is not currently a writer, then BtShared.isPending must
** be zero already. So this next line is harmless in that case.
*/
pBt.isPending = 0;
}
}

        static void downgradeAllSharedCacheTableLocks(Btree p)
        {
            BtShared pBt = p.pBt;
            if (pBt.pWriter == p)
            {
                BtLock pLock;
                pBt.pWriter = 0;
                pBt.isExclusive = 0;
                pBt.isPending = 0;
                for (pLock = pBt.pLock; pLock; pLock = pLock.pNext)
                {
                    Debug.Assert(pLock.eLock == READ_LOCK || pLock.pBtree == p);
                    pLock.eLock = READ_LOCK;
                }
            }
        }
#else
        internal static RC querySharedCacheTableLock(Btree p, Pgno iTab, BtreeLock.LOCK eLock) { return RC.OK; }
        internal static void clearAllSharedCacheTableLocks(Btree a) { }
        internal static void downgradeAllSharedCacheTableLocks(Btree a) { }
        internal static bool hasSharedCacheTableLock(Btree a, Pgno b, int c, int d) { return true; }
        internal static bool hasReadConflicts(Btree a, Pgno b) { return false; }
#endif
    }
}
