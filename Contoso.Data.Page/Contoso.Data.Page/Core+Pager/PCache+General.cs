using System;
using System.Diagnostics;
using Contoso.Sys;
using IPCache = Contoso.Core.Name.PCache1;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public partial class PCache
    {
        private const int N_SORT_BUCKET = 32;

#if SQLITE_ENABLE_EXPENSIVE_ASSERT
        private static void expensive_assert(bool x) { Debug.Assert(x); }
#endif

        internal static RC sqlite3PcacheInitialize()  {  return IPCache.xInit(IPCache.pArg);  }
        internal static void sqlite3PcacheShutdown()  { IPCache.xShutdown(IPCache.pArg); }
        internal static int sqlite3PcacheSize() { return 4; }

        internal static void sqlite3PcacheOpen(int szPage, int szExtra, bool bPurgeable, Func<object, PgHdr, RC> xStress, object pStress, PCache p)
        {
            p.Clear();
            p.szPage = szPage;
            p.szExtra = szExtra;
            p.bPurgeable = bPurgeable;
            p.xStress = xStress;
            p.pStress = pStress;
            p.nMax = 100;
        }

        internal void sqlite3PcacheSetPageSize(int szPage)
        {
            Debug.Assert(nRef == 0 && pDirty == null);
            if (pCache != null)
            {
                IPCache.xDestroy(ref pCache);
                pCache = null;
            }
            this.szPage = szPage;
        }

        internal RC sqlite3PcacheFetch(Pgno pgno, int createFlag, ref PgHdr ppPage)
        {
            Debug.Assert(createFlag == 1 || createFlag == 0);
            Debug.Assert(pgno > 0);
            // If the pluggable cache (sqlite3_pcache*) has not been allocated, allocate it now.
            if (null == pCache && createFlag != 0)
            {
                var nByte = pCache.szPage + szExtra + 0;
                var p = IPCache.xCreate(nByte, pCache.bPurgeable);
                p.xCachesize(pCache.nMax);
                pCache = p;
            }
            var eCreate = createFlag * (1 + ((!pCache.bPurgeable || null == pDirty) ? 1 : 0));
            PgHdr pPage = null;
            if (pCache != null)
                pPage = pCache.xFetch(pgno, eCreate);
            if (null == pPage && eCreate == 1)
            {
                PgHdr pPg;
                // Find a dirty page to write-out and recycle. First try to find a page that does not require a journal-sync (one with PGHDR_NEED_SYNC
                // cleared), but if that is not possible settle for any other unreferenced dirty page.
#if SQLITE_ENABLE_EXPENSIVE_ASSERT
                expensive_assert(pcacheCheckSynced(pCache));
#endif
                for (pPg = pSynced; pPg != null && (pPg.nRef != 0 || (pPg.flags & PgHdr.PGHDR.NEED_SYNC) != 0); pPg = pPg.pDirtyPrev) ;
                pSynced = pPg;
                if (null == pPg)
                    for (pPg = pDirtyTail; pPg != null && pPg.nRef != 0; pPg = pPg.pDirtyPrev) ;
                if (pPg != null)
                {
#if SQLITE_LOG_CACHE_SPILL
                    sqlite3_log(SQLITE_FULL, "spill page %d making room for %d - cache used: %d/%d", pPg.pgno, pgno, sqlite3GlobalConfig.pcache.xPagecount(pCache.pCache), pCache.nMax);
#endif
                    var rc = xStress(pStress, pPg);
                    if (rc != RC.OK && rc != RC.BUSY)
                        return rc;
                }
                pPage = pCache.xFetch(pgno, 2);
            }
            if (pPage != null)
            {
                if (null == pPage.pData)
                {
                    pPage.pData = MallocEx.sqlite3Malloc(pCache.szPage);
                    pPage.pCache = this;
                    pPage.pgno = pgno;
                }
                Debug.Assert(pPage.pCache == this);
                Debug.Assert(pPage.pgno == pgno);
                if (0 == pPage.nRef)
                    nRef++;
                pPage.nRef++;
                if (pgno == 1)
                    pPage1 = pPage;
            }
            ppPage = pPage;
            return (pPage == null && eCreate != 0 ? RC.NOMEM : RC.OK);
        }

        internal static void sqlite3PcacheRelease(PgHdr p)
        {
            Debug.Assert(p.nRef > 0);
            p.nRef--;
            if (p.nRef == 0)
            {
                var pCache = p.pCache;
                pCache.nRef--;
                if ((p.flags & PgHdr.PGHDR.DIRTY) == 0)
                    pcacheUnpin(p);
                else
                {
                    // Move the page to the head of the dirty list.
                    pcacheRemoveFromDirtyList(p);
                    pcacheAddToDirtyList(p);
                }
            }
        }

        internal static void sqlite3PcacheRef(PgHdr p) { Debug.Assert(p.nRef > 0); p.nRef++; }

        internal static void sqlite3PcacheDrop(PgHdr p)
        {
            PCache pCache;
            Debug.Assert(p.nRef == 1);
            if ((p.flags & PgHdr.PGHDR.DIRTY) != 0)
                pcacheRemoveFromDirtyList(p);
            pCache = p.pCache;
            pCache.nRef--;
            if (p.pgno == 1)
                pCache.pPage1 = null;
            pCache.pCache.xUnpin(p, true);
        }

        internal static void sqlite3PcacheMakeDirty(PgHdr p)
        {
            p.flags &= ~PgHdr.PGHDR.DONT_WRITE;
            Debug.Assert(p.nRef > 0);
            if (0 == (p.flags & PgHdr.PGHDR.DIRTY))
            {
                p.flags |= PgHdr.PGHDR.DIRTY;
                pcacheAddToDirtyList(p);
            }
        }

        internal static void sqlite3PcacheMakeClean(PgHdr p)
        {
            if ((p.flags & PgHdr.PGHDR.DIRTY) != 0)
            {
                pcacheRemoveFromDirtyList(p);
                p.flags &= ~(PgHdr.PGHDR.DIRTY | PgHdr.PGHDR.NEED_SYNC);
                if (p.nRef == 0)
                    pcacheUnpin(p);
            }
        }

        internal void sqlite3PcacheCleanAll()
        {
            PgHdr p;
            while ((p = pDirty) != null)
                sqlite3PcacheMakeClean(p);
        }

        internal void sqlite3PcacheClearSyncFlags()
        {
            for (var p = pDirty; p != null; p = p.pDirtyNext)
                p.flags &= ~PgHdr.PGHDR.NEED_SYNC;
            pSynced = pDirtyTail;
        }

        internal static void sqlite3PcacheMove(PgHdr p, Pgno newPgno)
        {
            PCache pCache = p.pCache;
            Debug.Assert(p.nRef > 0);
            Debug.Assert(newPgno > 0);
            pCache.pCache.xRekey(p, p.pgno, newPgno);
            p.pgno = newPgno;
            if ((p.flags & PgHdr.PGHDR.DIRTY) != 0 && (p.flags & PgHdr.PGHDR.NEED_SYNC) != 0)
            {
                pcacheRemoveFromDirtyList(p);
                pcacheAddToDirtyList(p);
            }
        }

        internal void sqlite3PcacheTruncate(Pgno pgno)
        {
            if (pCache != null)
            {
                PgHdr p;
                PgHdr pNext;
                for (p = pDirty; p != null; p = pNext)
                {
                    pNext = p.pDirtyNext;
                    // This routine never gets call with a positive pgno except right after sqlite3PcacheCleanAll().  So if there are dirty pages,
                    // it must be that pgno==0.
                    Debug.Assert(p.pgno > 0);
                    if (Check.ALWAYS(p.pgno > pgno))
                    {
                        Debug.Assert((p.flags & PgHdr.PGHDR.DIRTY) != 0);
                        sqlite3PcacheMakeClean(p);
                    }
                }
                if (pgno == 0 && pPage1 != null)
                {
                    pPage1.pData = MallocEx.sqlite3Malloc(szPage);
                    pgno = 1;
                }
                pCache.xTruncate(pgno + 1);
            }
        }

        internal void sqlite3PcacheClose()
        {
            if (pCache != null)
                IPCache.xDestroy(ref pCache);
        }

        internal void sqlite3PcacheClear() { sqlite3PcacheTruncate(0); }

        internal static PgHdr pcacheMergeDirtyList(PgHdr pA, PgHdr pB)
        {
            var result = new PgHdr();
            var pTail = result;
            while (pA != null && pB != null)
            {
                if (pA.pgno < pB.pgno)
                {
                    pTail.pDirty = pA;
                    pTail = pA;
                    pA = pA.pDirty;
                }
                else
                {
                    pTail.pDirty = pB;
                    pTail = pB;
                    pB = pB.pDirty;
                }
            }
            if (pA != null)
                pTail.pDirty = pA;
            else if (pB != null)
                pTail.pDirty = pB;
            else
                pTail.pDirty = null;
            return result.pDirty;
        }

        internal static PgHdr pcacheSortDirtyList(PgHdr pIn)
        {
            var a = new PgHdr[N_SORT_BUCKET];
            PgHdr p;
            while (pIn != null)
            {
                p = pIn;
                pIn = p.pDirty;
                p.pDirty = null;
                int i;
                for (i = 0; Check.ALWAYS(i < N_SORT_BUCKET - 1); i++)
                {
                    if (a[i] == null)
                    {
                        a[i] = p;
                        break;
                    }
                    else
                    {
                        p = pcacheMergeDirtyList(a[i], p);
                        a[i] = null;
                    }
                }
                if (Check.NEVER(i == N_SORT_BUCKET - 1))
                    // To get here, there need to be 2^(N_SORT_BUCKET) elements in the input list.  But that is impossible.
                    a[i] = pcacheMergeDirtyList(a[i], p);
            }
            p = a[0];
            for (var i = 1; i < N_SORT_BUCKET; i++)
                p = pcacheMergeDirtyList(p, a[i]);
            return p;
        }

        internal PgHdr sqlite3PcacheDirtyList()
        {
            for (var p = pDirty; p != null; p = p.pDirtyNext)
                p.pDirty = p.pDirtyNext;
            return pcacheSortDirtyList(pDirty);
        }

        internal int sqlite3PcacheRefCount() { return nRef; }
        internal static int sqlite3PcachePageRefcount(PgHdr p) { return p.nRef; }
        internal int sqlite3PcachePagecount() { return (pCache != null ? pCache.xPagecount() : 0); }
        internal void sqlite3PcacheSetCachesize(int mxPage)
        {
            nMax = mxPage;
            if (pCache != null)
                pCache.xCachesize(mxPage);
        }

#if DEBUG || SQLITE_CHECK_PAGES
        // For all dirty pages currently in the cache, invoke the specified callback. This is only used if the SQLITE_CHECK_PAGES macro is defined.
        internal void sqlite3PcacheIterateDirty(Action<PgHdr> xIter)
        {
            for (var pDirty = this.pDirty; pDirty != null; pDirty = pDirty.pDirtyNext)
                xIter(pDirty);
        }
#endif
    }
}
