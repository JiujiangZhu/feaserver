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
        internal SQLITE sqlite3BtreeCommitPhaseOne(string zMaster)
        {
            var rc = SQLITE.OK;
            if (this.inTrans == TRANS.WRITE)
            {
                var pBt = this.pBt;
                sqlite3BtreeEnter();
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum)
                {
                    rc = pBt.autoVacuumCommit();
                    if (rc != SQLITE.OK)
                    {
                        sqlite3BtreeLeave();
                        return rc;
                    }
                }
#endif
                rc = pBt.pPager.sqlite3PagerCommitPhaseOne(zMaster, false);
                sqlite3BtreeLeave();
            }
            return rc;
        }

        internal void btreeEndTransaction()
        {
            var pBt = this.pBt;
            Debug.Assert(sqlite3BtreeHoldsMutex(this));
            pBt.btreeClearHasContent();
            if (this.inTrans > TRANS.NONE && this.db.activeVdbeCnt > 1)
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
                    pBt.nTransaction--;
                    if (pBt.nTransaction == 0)
                        pBt.inTransaction = TRANS.NONE;
                }
                // Set the current transaction state to TRANS_NONE and unlock the pager if this call closed the only read or write transaction.
                this.inTrans = TRANS.NONE;
                pBt.unlockBtreeIfUnused();
            }
            btreeIntegrity();
        }

        internal SQLITE sqlite3BtreeCommitPhaseTwo(int bCleanup)
        {
            if (this.inTrans == TRANS.NONE)
                return SQLITE.OK;
            sqlite3BtreeEnter();
            btreeIntegrity();
            // If the handle has a write-transaction open, commit the shared-btrees transaction and set the shared state to TRANS_READ.
            if (this.inTrans == TRANS.WRITE)
            {
                var pBt = this.pBt;
                Debug.Assert(pBt.inTransaction == TRANS.WRITE);
                Debug.Assert(pBt.nTransaction > 0);
                var rc = pBt.pPager.sqlite3PagerCommitPhaseTwo();
                if (rc != SQLITE.OK && bCleanup == 0)
                {
                    sqlite3BtreeLeave();
                    return rc;
                }
                pBt.inTransaction = TRANS.READ;
            }
            btreeEndTransaction();
            sqlite3BtreeLeave();
            return SQLITE.OK;
        }

        internal SQLITE sqlite3BtreeCommit()
        {
            sqlite3BtreeEnter();
            var rc = sqlite3BtreeCommitPhaseOne(null);
            if (rc == SQLITE.OK)
                rc = sqlite3BtreeCommitPhaseTwo(0);
            sqlite3BtreeLeave();
            return rc;
        }

        internal void sqlite3BtreeTripAllCursors(int errCode)
        {
            sqlite3BtreeEnter();
            for (var p = this.pBt.pCursor; p != null; p = p.pNext)
            {
                p.sqlite3BtreeClearCursor();
                p.eState = CURSOR.FAULT;
                p.skipNext = errCode;
                for (var i = 0; i <= p.iPage; i++)
                {
                    p.apPage[i].releasePage();
                    p.apPage[i] = null;
                }
            }
            sqlite3BtreeLeave();
        }

        internal SQLITE sqlite3BtreeRollback()
        {
            sqlite3BtreeEnter();
            var pBt = this.pBt;
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
                Debug.Assert(pBt.inTransaction == TRANS.WRITE);
                var rc2 = pBt.pPager.sqlite3PagerRollback();
                if (rc2 != SQLITE.OK)
                    rc = rc2;
                // The rollback may have destroyed the pPage1.aData value.  So call btreeGetPage() on page 1 again to make
                // sure pPage1.aData is set correctly.
                var pPage1 = new MemPage();
                if (pBt.btreeGetPage(1, ref pPage1, 0) == SQLITE.OK)
                {
                    Pgno nPage = ConvertEx.sqlite3Get4byte(pPage1.aData, 28);
                    if (nPage == 0)
                        pBt.pPager.sqlite3PagerPagecount(out nPage);
                    pBt.nPage = nPage;
                    pPage1.releasePage();
                }
                Debug.Assert(pBt.countWriteCursors() == 0);
                pBt.inTransaction = TRANS.READ;
            }

            btreeEndTransaction();
            sqlite3BtreeLeave();
            return rc;
        }
    }
}
