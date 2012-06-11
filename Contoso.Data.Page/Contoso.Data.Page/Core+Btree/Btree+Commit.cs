using System.Diagnostics;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class Btree
    {
        internal RC sqlite3BtreeCommitPhaseOne(string zMaster)
        {
            var rc = RC.OK;
            if (this.inTrans == TRANS.WRITE)
            {
                var pBt = this.Shared;
                sqlite3BtreeEnter();
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.AutoVacuum)
                {
                    rc = MemPage.autoVacuumCommit(pBt);
                    if (rc != RC.OK)
                    {
                        sqlite3BtreeLeave();
                        return rc;
                    }
                }
#endif
                rc = pBt.Pager.sqlite3PagerCommitPhaseOne(zMaster, false);
                sqlite3BtreeLeave();
            }
            return rc;
        }

        internal void btreeEndTransaction()
        {
            var pBt = this.Shared;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            pBt.btreeClearHasContent();
            if (this.inTrans > TRANS.NONE && this.DB.activeVdbeCnt > 1)
            {
                // If there are other active statements that belong to this database handle, downgrade to a read-only transaction. The other statements
                // may still be reading from the database.
                downgradeAllSharedCacheTableLocks(this);
                this.inTrans = TRANS.READ;
            }
            else
            {
                // If the handle had any kind of transaction open, decrement the transaction count of the shared btree. If the transaction count
                // reaches 0, set the shared state to TRANS_NONE. The unlockBtreeIfUnused() call below will unlock the pager.  */
                if (this.inTrans != TRANS.NONE)
                {
                    clearAllSharedCacheTableLocks(this);
                    pBt.Transactions--;
                    if (pBt.Transactions == 0)
                        pBt.InTransaction = TRANS.NONE;
                }
                // Set the current transaction state to TRANS_NONE and unlock the pager if this call closed the only read or write transaction.
                this.inTrans = TRANS.NONE;
                pBt.unlockBtreeIfUnused();
            }
            btreeIntegrity();
        }

        internal RC sqlite3BtreeCommitPhaseTwo(int bCleanup)
        {
            if (this.inTrans == TRANS.NONE)
                return RC.OK;
            sqlite3BtreeEnter();
            btreeIntegrity();
            // If the handle has a write-transaction open, commit the shared-btrees transaction and set the shared state to TRANS_READ.
            if (this.inTrans == TRANS.WRITE)
            {
                var pBt = this.Shared;
                Debug.Assert(pBt.InTransaction == TRANS.WRITE);
                Debug.Assert(pBt.Transactions > 0);
                var rc = pBt.Pager.sqlite3PagerCommitPhaseTwo();
                if (rc != RC.OK && bCleanup == 0)
                {
                    sqlite3BtreeLeave();
                    return rc;
                }
                pBt.InTransaction = TRANS.READ;
            }
            btreeEndTransaction();
            sqlite3BtreeLeave();
            return RC.OK;
        }

        internal RC sqlite3BtreeCommit()
        {
            sqlite3BtreeEnter();
            var rc = sqlite3BtreeCommitPhaseOne(null);
            if (rc == RC.OK)
                rc = sqlite3BtreeCommitPhaseTwo(0);
            sqlite3BtreeLeave();
            return rc;
        }

        internal void sqlite3BtreeTripAllCursors(int errCode)
        {
            sqlite3BtreeEnter();
            for (var p = this.Shared.Cursors; p != null; p = p.Next)
            {
                p.Clear();
                p.State = CURSOR.FAULT;
                p.SkipNext = errCode;
                for (var i = 0; i <= p.PageID; i++)
                {
                    p.Pages[i].releasePage();
                    p.Pages[i] = null;
                }
            }
            sqlite3BtreeLeave();
        }

        internal RC sqlite3BtreeRollback()
        {
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            var rc = pBt.saveAllCursors(0, null);
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
            btreeIntegrity();
            if (this.inTrans == TRANS.WRITE)
            {
                Debug.Assert(pBt.InTransaction == TRANS.WRITE);
                var rc2 = pBt.Pager.sqlite3PagerRollback();
                if (rc2 != RC.OK)
                    rc = rc2;
                // The rollback may have destroyed the pPage1.aData value.  So call btreeGetPage() on page 1 again to make
                // sure pPage1.aData is set correctly.
                var pPage1 = new MemPage();
                if (pBt.btreeGetPage(1, ref pPage1, 0) == RC.OK)
                {
                    Pgno nPage = ConvertEx.Get4(pPage1.Data, 28);
                    if (nPage == 0)
                        pBt.Pager.sqlite3PagerPagecount(out nPage);
                    pBt.Pages = nPage;
                    pPage1.releasePage();
                }
                Debug.Assert(pBt.countWriteCursors() == 0);
                pBt.InTransaction = TRANS.READ;
            }

            btreeEndTransaction();
            sqlite3BtreeLeave();
            return rc;
        }
    }
}
