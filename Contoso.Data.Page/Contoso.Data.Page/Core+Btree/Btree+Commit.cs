using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;
namespace Contoso.Core
{
    public partial class Btree
    {
        internal static SQLITE sqlite3BtreeCommitPhaseOne(Btree p, string zMaster)
        {
            var rc = SQLITE.OK;
            if (p.inTrans == TRANS.WRITE)
            {
                var pBt = p.pBt;
                sqlite3BtreeEnter(p);
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum)
                {
                    rc = autoVacuumCommit(pBt);
                    if (rc != SQLITE.OK)
                    {
                        sqlite3BtreeLeave(p);
                        return rc;
                    }
                }
#endif
                rc = pBt.pPager.sqlite3PagerCommitPhaseOne(zMaster, false);
                sqlite3BtreeLeave(p);
            }
            return rc;
        }

        internal static void btreeEndTransaction(Btree p)
        {
            var pBt = p.pBt;
            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            btreeClearHasContent(pBt);
            if (p.inTrans > TRANS.NONE && p.db.activeVdbeCnt > 1)
            {
                // If there are other active statements that belong to this database handle, downgrade to a read-only transaction. The other statements
                // may still be reading from the database.
                downgradeAllSharedCacheTableLocks(p);
                p.inTrans = TRANS.READ;
            }
            else
            {
                // If the handle had any kind of transaction open, decrement the transaction count of the shared btree. If the transaction count
                // reaches 0, set the shared state to TRANS_NONE. The unlockBtreeIfUnused() call below will unlock the pager.  */
                if (p.inTrans != TRANS.NONE)
                {
                    clearAllSharedCacheTableLocks(p);
                    pBt.nTransaction--;
                    if (pBt.nTransaction == 0)
                        pBt.inTransaction = TRANS.NONE;
                }
                // Set the current transaction state to TRANS_NONE and unlock the pager if this call closed the only read or write transaction.
                p.inTrans = TRANS.NONE;
                unlockBtreeIfUnused(pBt);
            }
            btreeIntegrity(p);
        }

        internal static SQLITE sqlite3BtreeCommitPhaseTwo(Btree p, int bCleanup)
        {
            if (p.inTrans == TRANS.NONE)
                return SQLITE.OK;
            sqlite3BtreeEnter(p);
            btreeIntegrity(p);
            // If the handle has a write-transaction open, commit the shared-btrees transaction and set the shared state to TRANS_READ.
            if (p.inTrans == TRANS.WRITE)
            {
                var pBt = p.pBt;
                Debug.Assert(pBt.inTransaction == TRANS.WRITE);
                Debug.Assert(pBt.nTransaction > 0);
                var rc = pBt.pPager.sqlite3PagerCommitPhaseTwo();
                if (rc != SQLITE.OK && bCleanup == 0)
                {
                    sqlite3BtreeLeave(p);
                    return rc;
                }
                pBt.inTransaction = TRANS.READ;
            }
            btreeEndTransaction(p);
            sqlite3BtreeLeave(p);
            return SQLITE.OK;
        }

        internal static SQLITE sqlite3BtreeCommit(Btree p)
        {
            sqlite3BtreeEnter(p);
            var rc = sqlite3BtreeCommitPhaseOne(p, null);
            if (rc == SQLITE.OK)
                rc = sqlite3BtreeCommitPhaseTwo(p, 0);
            sqlite3BtreeLeave(p);
            return rc;
        }

#if DEBUG
        internal static int countWriteCursors(BtShared pBt)
        {
            var r = 0;
            for (var pCur = pBt.pCursor; pCur != null; pCur = pCur.pNext)
                if (pCur.wrFlag != 0 && pCur.eState != CURSOR.FAULT)
                    r++;
            return r;
        }
#else
        internal static int countWriteCursors(BtShared pBt) { return -1; }
#endif

        static void sqlite3BtreeTripAllCursors(Btree pBtree, int errCode)
        {
            sqlite3BtreeEnter(pBtree);
            for (var p = pBtree.pBt.pCursor; p != null; p = p.pNext)
            {
                sqlite3BtreeClearCursor(p);
                p.eState = CURSOR.FAULT;
                p.skipNext = errCode;
                for (var i = 0; i <= p.iPage; i++)
                {
                    releasePage(p.apPage[i]);
                    p.apPage[i] = null;
                }
            }
            sqlite3BtreeLeave(pBtree);
        }

        internal static SQLITE sqlite3BtreeRollback(Btree p)
        {
            sqlite3BtreeEnter(p);
            var pBt = p.pBt;
            var rc = saveAllCursors(pBt, 0, null);
#if !SQLITE_OMIT_SHARED_CACHE
if( rc!=SQLITE_OK ){
/* This is a horrible situation. An IO or malloc() error occurred whilst
** trying to save cursor positions. If this is an automatic rollback (as
** the result of a constraint, malloc() failure or IO error) then
** the cache may be internally inconsistent (not contain valid trees) so
** we cannot simply return the error to the caller. Instead, abort
** all queries that may be using any of the cursors that failed to save.
*/
sqlite3BtreeTripAllCursors(p, rc);
}
#endif
            btreeIntegrity(p);
            if (p.inTrans == TRANS.WRITE)
            {
                Debug.Assert(pBt.inTransaction == TRANS.WRITE);
                var rc2 = pBt.pPager.sqlite3PagerRollback();
                if (rc2 != SQLITE.OK)
                    rc = rc2;
                // The rollback may have destroyed the pPage1.aData value.  So call btreeGetPage() on page 1 again to make
                // sure pPage1.aData is set correctly.
                var pPage1 = new MemPage();
                if (btreeGetPage(pBt, 1, ref pPage1, 0) == SQLITE.OK)
                {
                    Pgno nPage = ConvertEx.sqlite3Get4byte(pPage1.aData, 28);
                    if (nPage == 0)
                        pBt.pPager.sqlite3PagerPagecount(out nPage);
                    pBt.nPage = nPage;
                    releasePage(pPage1);
                }
                Debug.Assert(countWriteCursors(pBt) == 0);
                pBt.inTransaction = TRANS.READ;
            }

            btreeEndTransaction(p);
            sqlite3BtreeLeave(p);
            return rc;
        }

    }
}
