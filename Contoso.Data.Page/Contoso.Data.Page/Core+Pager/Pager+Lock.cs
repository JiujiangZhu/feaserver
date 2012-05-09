using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
namespace Contoso.Core
{
    public partial class Pager
    {
        // Unlock the database file to level eLock, which must be either NO_LOCK or SHARED_LOCK. Regardless of whether or not the call to xUnlock()
        // succeeds, set the Pager.eLock variable to match the (attempted) new lock.
        // Except, if Pager.eLock is set to UNKNOWN_LOCK when this function is called, do not modify it. See the comment above the #define of 
        // UNKNOWN_LOCK for an explanation of this.
        static SQLITE pagerUnlockDb(Pager pPager, LOCK eLock)
        {
            var rc = SQLITE.OK;
            Debug.Assert(!pPager.exclusiveMode || pPager.eLock == eLock);
            Debug.Assert(eLock == LOCK.NO || eLock == LOCK.SHARED);
            Debug.Assert(eLock != LOCK.NO || !pagerUseWal(pPager));
            if (pPager.fd.isOpen)
            {
                Debug.Assert(pPager.eLock >= eLock);
                rc = sqlite3OsUnlock(pPager.fd, eLock);
                if (pPager.eLock != LOCK.UNKNOWN)
                    pPager.eLock = eLock;
                IOTRACE("UNLOCK %p %d\n", pPager, eLock);
            }
            return rc;
        }

        // Lock the database file to level eLock, which must be either SHARED_LOCK, RESERVED_LOCK or EXCLUSIVE_LOCK. If the caller is successful, set the
        // Pager.eLock variable to the new locking state. 
        //
        // Except, if Pager.eLock is set to UNKNOWN_LOCK when this function is called, do not modify it unless the new locking state is EXCLUSIVE_LOCK. 
        // See the comment above the #define of UNKNOWN_LOCK for an explanation of this.
        static SQLITE pagerLockDb(Pager pPager, LOCK eLock)
        {
            var rc = SQLITE.OK;
            Debug.Assert(eLock == LOCK.SHARED || eLock == LOCK.RESERVED || eLock == LOCK.EXCLUSIVE);
            if (pPager.eLock < eLock || pPager.eLock == LOCK.UNKNOWN)
            {
                rc = sqlite3OsLock(pPager.fd, eLock);
                if (rc == SQLITE.OK && (pPager.eLock != LOCK.UNKNOWN || eLock == LOCK.EXCLUSIVE))
                {
                    pPager.eLock = (u8)eLock;
                    IOTRACE("LOCK %p %d\n", pPager, eLock);
                }
            }
            return rc;
        }
    }
}
