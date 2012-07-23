using System;
using System.Diagnostics;
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

        private static RC sqlite3PcacheInitialize() { return IPCache.xInit(IPCache.pArg); }
        private static void sqlite3PcacheShutdown() { IPCache.xShutdown(IPCache.pArg); }
        internal static int sqlite3PcacheSize() { return 4; }

        // was:sqlite3PcacheOpen
        internal static void Open(int szPage, int szExtra, bool bPurgeable, Func<object, PgHdr, RC> xStress, object pStress, PCache p)
        {
            p.ClearState();
            p.szPage = szPage;
            p.szExtra = szExtra;
            p.bPurgeable = bPurgeable;
            p.xStress = xStress;
            p.pStress = pStress;
            p.nMax = 100;
        }

        // was:sqlite3PcacheSetPageSize
        internal void SetPageSize(int szPage)
        {
            Debug.Assert(nRef == 0 && pDirty == null);
            if (pCache != null)
            {
                IPCache.xDestroy(ref pCache);
                pCache = null;
            }
            this.szPage = szPage;
        }

        // was:sqlite3PcacheFetch
        internal RC FetchPage(Pgno pgno, int createFlag, ref PgHdr ppPage)
        {
            Debug.Assert(createFlag == 1 || createFlag == 0);
            Debug.Assert(pgno > 0);
            // If the pluggable cache (sqlite3_pcache*) has not been allocated, allocate it now.
            if (pCache == null && createFlag != 0)
            {
                var nByte = szPage + szExtra + 0;
                var p = IPCache.xCreate(nByte, bPurgeable);
                p.xCachesize(nMax);
                pCache = p;
            }
            var eCreate = createFlag * (1 + ((!bPurgeable || null == pDirty) ? 1 : 0));
            PgHdr pPage = null;
            if (pCache != null)
                pPage = pCache.xFetch(pgno, eCreate);
            if (pPage == null && eCreate == 1)
            {
                PgHdr pPg;
                // Find a dirty page to write-out and recycle. First try to find a page that does not require a journal-sync (one with PGHDR_NEED_SYNC
                // cleared), but if that is not possible settle for any other unreferenced dirty page.
#if SQLITE_ENABLE_EXPENSIVE_ASSERT
                expensive_assert(pcacheCheckSynced(pCache));
#endif
                for (pPg = pSynced; pPg != null && (pPg.Refs != 0 || (pPg.Flags & PgHdr.PGHDR.NEED_SYNC) != 0); pPg = pPg.DirtyPrev) ;
                pSynced = pPg;
                if (pPg == null)
                    for (pPg = pDirtyTail; pPg != null && pPg.Refs != 0; pPg = pPg.DirtyPrev) ;
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
                if (pPage.Data == null)
                {
                    pPage.Data = MallocEx.sqlite3Malloc(pCache.szPage);
                    pPage.Cache = this;
                    pPage.ID = pgno;
                }
                Debug.Assert(pPage.Cache == this);
                Debug.Assert(pPage.ID == pgno);
                if (pPage.Refs == 0)
                    nRef++;
                pPage.Refs++;
                if (pgno == 1)
                    pPage1 = pPage;
            }
            ppPage = pPage;
            return (pPage == null && eCreate != 0 ? RC.NOMEM : RC.OK);
        }

        // was:sqlite3PcacheRelease
        internal static void ReleasePage(PgHdr p)
        {
            Debug.Assert(p.Refs > 0);
            p.Refs--;
            if (p.Refs == 0)
            {
                var pCache = p.Cache;
                pCache.nRef--;
                if ((p.Flags & PgHdr.PGHDR.DIRTY) == 0)
                    pcacheUnpin(p);
                else
                {
                    // Move the page to the head of the dirty list.
                    pcacheRemoveFromDirtyList(p);
                    pcacheAddToDirtyList(p);
                }
            }
        }

        // was:sqlite3PcacheRef
        internal static void AddPageRef(PgHdr p) { Debug.Assert(p.Refs > 0); p.Refs++; }

        // was:sqlite3PcacheDrop
        internal static void DropPage(PgHdr p)
        {
            PCache pCache;
            Debug.Assert(p.Refs == 1);
            if ((p.Flags & PgHdr.PGHDR.DIRTY) != 0)
                pcacheRemoveFromDirtyList(p);
            pCache = p.Cache;
            pCache.nRef--;
            if (p.ID == 1)
                pCache.pPage1 = null;
            pCache.pCache.xUnpin(p, true);
        }

        // was:sqlite3PcacheMakeDirty
        internal static void MakePageDirty(PgHdr p)
        {
            p.Flags &= ~PgHdr.PGHDR.DONT_WRITE;
            Debug.Assert(p.Refs > 0);
            if (0 == (p.Flags & PgHdr.PGHDR.DIRTY))
            {
                p.Flags |= PgHdr.PGHDR.DIRTY;
                pcacheAddToDirtyList(p);
            }
        }

        // was:sqlite3PcacheMakeClean
        internal static void MakePageClean(PgHdr p)
        {
            if ((p.Flags & PgHdr.PGHDR.DIRTY) != 0)
            {
                pcacheRemoveFromDirtyList(p);
                p.Flags &= ~(PgHdr.PGHDR.DIRTY | PgHdr.PGHDR.NEED_SYNC);
                if (p.Refs == 0)
                    pcacheUnpin(p);
            }
        }

        // was:sqlite3PcacheCleanAll
        internal void CleanAllPages()
        {
            PgHdr p;
            while ((p = pDirty) != null)
                MakePageClean(p);
        }

        // was:sqlite3PcacheClearSyncFlags
        internal void ClearSyncFlags()
        {
            for (var p = pDirty; p != null; p = p.DirtyNext)
                p.Flags &= ~PgHdr.PGHDR.NEED_SYNC;
            pSynced = pDirtyTail;
        }

        // was:sqlite3PcacheMove
        internal static void MovePage(PgHdr p, Pgno newPgno)
        {
            PCache pCache = p.Cache;
            Debug.Assert(p.Refs > 0);
            Debug.Assert(newPgno > 0);
            pCache.pCache.xRekey(p, p.ID, newPgno);
            p.ID = newPgno;
            if ((p.Flags & PgHdr.PGHDR.DIRTY) != 0 && (p.Flags & PgHdr.PGHDR.NEED_SYNC) != 0)
            {
                pcacheRemoveFromDirtyList(p);
                pcacheAddToDirtyList(p);
            }
        }

        // was:sqlite3PcacheTruncate
        internal void TruncatePage(Pgno pgno)
        {
            if (pCache != null)
            {
                PgHdr p;
                PgHdr pNext;
                for (p = pDirty; p != null; p = pNext)
                {
                    pNext = p.DirtyNext;
                    // This routine never gets call with a positive pgno except right after sqlite3PcacheCleanAll().  So if there are dirty pages,
                    // it must be that pgno==0.
                    Debug.Assert(p.ID > 0);
                    if (Check.ALWAYS(p.ID > pgno))
                    {
                        Debug.Assert((p.Flags & PgHdr.PGHDR.DIRTY) != 0);
                        MakePageClean(p);
                    }
                }
                if (pgno == 0 && pPage1 != null)
                {
                    pPage1.Data = MallocEx.sqlite3Malloc(szPage);
                    pgno = 1;
                }
                pCache.xTruncate(pgno + 1);
            }
        }

        // was:sqlite3PcacheClose
        internal void Close()
        {
            if (pCache != null)
                IPCache.xDestroy(ref pCache);
        }

        // was:sqlite3PcacheClear
        internal void Clear() { TruncatePage(0); }

        private static PgHdr pcacheMergeDirtyList(PgHdr pA, PgHdr pB)
        {
            var result = new PgHdr();
            var pTail = result;
            while (pA != null && pB != null)
            {
                if (pA.ID < pB.ID)
                {
                    pTail.Dirtys = pA;
                    pTail = pA;
                    pA = pA.Dirtys;
                }
                else
                {
                    pTail.Dirtys = pB;
                    pTail = pB;
                    pB = pB.Dirtys;
                }
            }
            if (pA != null)
                pTail.Dirtys = pA;
            else if (pB != null)
                pTail.Dirtys = pB;
            else
                pTail.Dirtys = null;
            return result.Dirtys;
        }

        private static PgHdr pcacheSortDirtyList(PgHdr pIn)
        {
            var a = new PgHdr[N_SORT_BUCKET];
            PgHdr p;
            while (pIn != null)
            {
                p = pIn;
                pIn = p.Dirtys;
                p.Dirtys = null;
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
            for (var p = pDirty; p != null; p = p.DirtyNext)
                p.Dirtys = p.DirtyNext;
            return pcacheSortDirtyList(pDirty);
        }

        internal int sqlite3PcacheRefCount() { return nRef; }
        internal static int sqlite3PcachePageRefcount(PgHdr p) { return p.Refs; }
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
            for (var pDirty = this.pDirty; pDirty != null; pDirty = pDirty.DirtyNext)
                xIter(pDirty);
        }
#endif
    }
}
