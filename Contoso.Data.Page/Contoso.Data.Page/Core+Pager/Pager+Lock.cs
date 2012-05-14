using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using LOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal static void pagerUnlockIfUnused(Pager pPager)
        {
            if (pPager.pPCache.sqlite3PcacheRefCount() == 0)
                pagerUnlockAndRollback(pPager);
        }

        internal static void pagerUnlockAndRollback(Pager pPager)
        {
            if (pPager.eState != PAGER.ERROR && pPager.eState != PAGER.OPEN)
            {
                Debug.Assert(assert_pager_state(pPager));
                if (pPager.eState >= PAGER.WRITER_LOCKED)
                {
                    MallocEx.sqlite3BeginBenignMalloc();
                    sqlite3PagerRollback(pPager);
                    MallocEx.sqlite3EndBenignMalloc();
                }
                else if (!pPager.exclusiveMode)
                {
                    Debug.Assert(pPager.eState == PAGER.READER);
                    pager_end_transaction(pPager, 0);
                }
            }
            pager_unlock(pPager);
        }

        internal static SQLITE pagerUnlockDb(Pager pPager, LOCK eLock)
        {
            var rc = SQLITE.OK;
            Debug.Assert(!pPager.exclusiveMode || pPager.eLock == eLock);
            Debug.Assert(eLock == LOCK.NO || eLock == LOCK.SHARED);
            Debug.Assert(eLock != LOCK.NO || !pPager.pagerUseWal());
            if (pPager.fd.isOpen)
            {
                Debug.Assert(pPager.eLock >= eLock);
                rc = FileEx.sqlite3OsUnlock(pPager.fd, eLock);
                if (pPager.eLock != LOCK.UNKNOWN)
                    pPager.eLock = eLock;
                SysEx.IOTRACE("UNLOCK %p %d\n", pPager, eLock);
            }
            return rc;
        }

        internal static SQLITE pagerLockDb(Pager pPager, LOCK eLock)
        {
            var rc = SQLITE.OK;
            Debug.Assert(eLock == LOCK.SHARED || eLock == LOCK.RESERVED || eLock == LOCK.EXCLUSIVE);
            if (pPager.eLock < eLock || pPager.eLock == LOCK.UNKNOWN)
            {
                rc = FileEx.sqlite3OsLock(pPager.fd, eLock);
                if (rc == SQLITE.OK && (pPager.eLock != LOCK.UNKNOWN || eLock == LOCK.EXCLUSIVE))
                {
                    pPager.eLock = eLock;
                    SysEx.IOTRACE("LOCK %p %d\n", pPager, eLock);
                }
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerExclusiveLock(Pager pPager)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD || pPager.eState == PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(pPager));
            if (!pagerUseWal(pPager))
                rc = pager_wait_on_lock(pPager, LOCK.EXCLUSIVE);
            return rc;
        }
    }
}
