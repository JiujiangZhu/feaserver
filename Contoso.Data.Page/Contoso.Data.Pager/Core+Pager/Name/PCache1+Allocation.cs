using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using System.Text;
using Contoso.Threading;
namespace Contoso.Core.Name
{
    public partial class PCache1
    {
        private static void sqlite3PCacheBufferSetup(object pBuf, int sz, int n)
        {
            if (pcache1 != null)
            {
                sz = SysEx.ROUNDDOWN8(sz);
                pcache1.szSlot = sz;
                pcache1.nSlot = pcache1.nFreeSlot = n;
                pcache1.nReserve = n > 90 ? 10 : (n / 10 + 1);
                pcache1.pStart = null;
                pcache1.pEnd = null;
                pcache1.pFree = null;
                pcache1.bUnderPressure = false;
                PgFreeslot p;
                while (n-- > 0)
                {
                    p = new PgFreeslot();
                    p._PgHdr = new PgHdr();
                    p.pNext = pcache1.pFree;
                    pcache1.pFree = p;
                }
                pcache1.pEnd = pBuf;
            }
        }

        private static PgHdr pcache1Alloc(int nByte)
        {
            PgHdr p = null;
            Debug.Assert(MutexEx.sqlite3_mutex_notheld(pcache1.grp.mutex));
            StatusEx.sqlite3StatusSet(StatusEx.STATUS.PAGECACHE_SIZE, nByte);
            if (nByte <= pcache1.szSlot)
            {
                MutexEx.sqlite3_mutex_enter(pcache1.mutex);
                p = pcache1.pFree._PgHdr;
                if (p != null)
                {
                    pcache1.pFree = pcache1.pFree.pNext;
                    pcache1.nFreeSlot--;
                    pcache1.bUnderPressure = pcache1.nFreeSlot < pcache1.nReserve;
                    Debug.Assert(pcache1.nFreeSlot >= 0);
                    StatusEx.sqlite3StatusAdd(StatusEx.STATUS.PAGECACHE_USED, 1);
                }
                MutexEx.sqlite3_mutex_leave(pcache1.mutex);
            }
            if (p == null)
            {
                // Memory is not available in the SQLITE_CONFIG_PAGECACHE pool.  Get it from sqlite3Malloc instead.
                p = new PgHdr();
                {
                    var sz = nByte;
                    MutexEx.sqlite3_mutex_enter(pcache1.mutex);
                    StatusEx.sqlite3StatusAdd(StatusEx.STATUS.PAGECACHE_OVERFLOW, sz);
                    MutexEx.sqlite3_mutex_leave(pcache1.mutex);
                }
                SysEx.sqlite3MemdebugSetType(p, SysEx.MEMTYPE.PCACHE);
            }
            return p;
        }

        private static void pcache1Free(ref PgHdr p)
        {
            if (p == null)
                return;
            if (p.CacheAllocated)
            {
                var pSlot = new PgFreeslot();
                MutexEx.sqlite3_mutex_enter(pcache1.mutex);
                StatusEx.sqlite3StatusAdd(StatusEx.STATUS.PAGECACHE_USED, -1);
                pSlot._PgHdr = p;
                pSlot.pNext = pcache1.pFree;
                pcache1.pFree = pSlot;
                pcache1.nFreeSlot++;
                pcache1.bUnderPressure = pcache1.nFreeSlot < pcache1.nReserve;
                Debug.Assert(pcache1.nFreeSlot <= pcache1.nSlot);
                MutexEx.sqlite3_mutex_leave(pcache1.mutex);
            }
            else
            {
                Debug.Assert(SysEx.sqlite3MemdebugHasType(p, SysEx.MEMTYPE.PCACHE));
                SysEx.sqlite3MemdebugSetType(p, SysEx.MEMTYPE.HEAP);
                var iSize = MallocEx.sqlite3MallocSize(p.Data);
                MutexEx.sqlite3_mutex_enter(pcache1.mutex);
                StatusEx.sqlite3StatusAdd(StatusEx.STATUS.PAGECACHE_OVERFLOW, -iSize);
                MutexEx.sqlite3_mutex_leave(pcache1.mutex);
                MallocEx.sqlite3_free(ref p.Data);
            }
        }

#if SQLITE_ENABLE_MEMORY_MANAGEMENT
        private static int pcache1MemSize(object p)
        {
            if (p >= pcache1.pStart && p < pcache1.pEnd)
                return pcache1.szSlot;
            else
            {
                Debug.Assert(SysEx.sqlite3MemdebugHasType(p, SysEx.MEMTYPE.PCACHE));
                SysEx.sqlite3MemdebugSetType(p, SysEx.MEMTYPE.HEAP);
                var iSize = SysEx.sqlite3MallocSize(p);
                SysEx.sqlite3MemdebugSetType(p, SysEx.MEMTYPE.PCACHE);
                return iSize;
            }
        }
#endif

        private PgHdr1 pcache1AllocPage()
        {
            var pPg = pcache1Alloc(szPage);
            var p = new PgHdr1();
            p.pCache = this;
            p.pPgHdr = pPg;
            if (bPurgeable)
                pGroup.nCurrentPage++;
            return p;
        }

        private static void pcache1FreePage(ref PgHdr1 p)
        {
            if (Check.ALWAYS(p) != null)
            {
                var pCache = p.pCache;
                if (pCache.bPurgeable)
                    pCache.pGroup.nCurrentPage--;
                pcache1Free(ref p.pPgHdr);
            }
        }

        private static PgHdr sqlite3PageMalloc(int sz) { return pcache1Alloc(sz); }

        public static void sqlite3PageFree(ref byte[] p)
        {
            if (p != null)
            {
                MallocEx.sqlite3_free(ref p);
                p = null;
            }
        }
        public static void sqlite3PageFree(ref PgHdr p) { pcache1Free(ref p); }

        private bool pcache1UnderMemoryPressure() { return (pcache1.nSlot != 0 && szPage <= pcache1.szSlot ? pcache1.bUnderPressure : MallocEx.sqlite3HeapNearlyFull()); }
    }
}
