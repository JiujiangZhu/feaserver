using System.Diagnostics;
using Contoso.Sys;
using LOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal void pagerUnlockIfUnused()
        {
            if (this.pPCache.sqlite3PcacheRefCount() == 0)
                pagerUnlockAndRollback();
        }

        internal void pagerUnlockAndRollback()
        {
            if (this.eState != PAGER.ERROR && this.eState != PAGER.OPEN)
            {
                Debug.Assert(assert_pager_state());
                if (this.eState >= PAGER.WRITER_LOCKED)
                {
                    MallocEx.sqlite3BeginBenignMalloc();
                    sqlite3PagerRollback();
                    MallocEx.sqlite3EndBenignMalloc();
                }
                else if (!this.exclusiveMode)
                {
                    Debug.Assert(this.eState == PAGER.READER);
                    pager_end_transaction(0);
                }
            }
            pager_unlock();
        }

        internal SQLITE pagerUnlockDb(LOCK eLock)
        {
            var rc = SQLITE.OK;
            Debug.Assert(!this.exclusiveMode || this.eLock == eLock);
            Debug.Assert(eLock == LOCK.NO || eLock == LOCK.SHARED);
            Debug.Assert(eLock != LOCK.NO || !this.pagerUseWal());
            if (this.fd.isOpen)
            {
                Debug.Assert(this.eLock >= eLock);
                rc = FileEx.sqlite3OsUnlock(this.fd, eLock);
                if (this.eLock != LOCK.UNKNOWN)
                    this.eLock = eLock;
                SysEx.IOTRACE("UNLOCK %p %d\n", this, eLock);
            }
            return rc;
        }

        internal SQLITE pagerLockDb(LOCK eLock)
        {
            var rc = SQLITE.OK;
            Debug.Assert(eLock == LOCK.SHARED || eLock == LOCK.RESERVED || eLock == LOCK.EXCLUSIVE);
            if (this.eLock < eLock || this.eLock == LOCK.UNKNOWN)
            {
                rc = FileEx.sqlite3OsLock(this.fd, eLock);
                if (rc == SQLITE.OK && (this.eLock != LOCK.UNKNOWN || eLock == LOCK.EXCLUSIVE))
                {
                    this.eLock = eLock;
                    SysEx.IOTRACE("LOCK %p %d\n", this, eLock);
                }
            }
            return rc;
        }

        internal SQLITE sqlite3PagerExclusiveLock()
        {
            var rc = SQLITE.OK;
            Debug.Assert(this.eState == PAGER.WRITER_CACHEMOD || this.eState == PAGER.WRITER_DBMOD || this.eState == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state());
            if (!pagerUseWal())
                rc = pager_wait_on_lock(LOCK.EXCLUSIVE);
            return rc;
        }
    }
}
