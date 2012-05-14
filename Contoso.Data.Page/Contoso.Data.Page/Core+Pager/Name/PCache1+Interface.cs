using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using System.Text;
using IPCache = Contoso.Core.Name.PCache1;
namespace Contoso.Core.Name
{
    public partial class PCache1
    {
        private static bool sqlite3GlobalConfig_bCoreMutex = false;
        private static PCacheGlobal pcache1;

        public static object pArg
        {
            get { return null; }
        }

        public static SQLITE xInit(object NotUsed)
        {
            SysEx.UNUSED_PARAMETER(NotUsed);
            Debug.Assert(!pcache1.isInit);
            pcache1 = new PCacheGlobal();
            if (sqlite3GlobalConfig_bCoreMutex)
            {
                pcache1.grp.mutex = MutexEx.sqlite3_mutex_alloc(MutexEx.MUTEX.STATIC_LRU);
                pcache1.mutex = MutexEx.sqlite3_mutex_alloc(MutexEx.MUTEX.STATIC_PMEM);
            }
            pcache1.grp.mxPinned = 10;
            pcache1.isInit = true;
            return SQLITE.OK;
        }

        public static void xShutdown(object NotUsed)
        {
            SysEx.UNUSED_PARAMETER(NotUsed);
            Debug.Assert(pcache1.isInit);
            pcache1 = new PCacheGlobal();
        }

        public static IPCache xCreate(int szPage, bool bPurgeable)
        {
            // The seperateCache variable is true if each PCache has its own private PGroup.  In other words, separateCache is true for mode (1) where no
            // mutexing is required.
            //   *  Always use a unified cache (mode-2) if ENABLE_MEMORY_MANAGEMENT
            //   *  Always use a unified cache in single-threaded applications
            //   *  Otherwise (if multi-threaded and ENABLE_MEMORY_MANAGEMENT is off) use separate caches (mode-1)
#if SQLITE_ENABLE_MEMORY_MANAGEMENT || !SQLITE_THREADSAF
            const int separateCache = 0;
#else
            int separateCache = sqlite3GlobalConfig.bCoreMutex > 0;
#endif
            var pCache = new PCache1();
            {
                PGroup pGroup;       // The group the new page cache will belong to
                if (separateCache != 0)
                {
                    //pGroup = new PGroup();
                    //pGroup.mxPinned = 10;
                }
                else
                    pGroup = pcache1.grp;
                pCache.pGroup = pGroup;
                pCache.szPage = szPage;
                pCache.bPurgeable = bPurgeable;
                if (bPurgeable)
                {
                    pCache.nMin = 10;
                    pcache1EnterMutex(pGroup);
                    pGroup.nMinPage += (int)pCache.nMin;
                    pGroup.mxPinned = pGroup.nMaxPage + 10 - pGroup.nMinPage;
                    pcache1LeaveMutex(pGroup);
                }
            }
            return (IPCache)pCache;
        }

        public void xCachesize(int nCachesize)
        {
            if (bPurgeable)
            {
                var pGroup = this.pGroup;
                pcache1EnterMutex(pGroup);
                pGroup.nMaxPage += nCachesize - nMax;
                pGroup.mxPinned = pGroup.nMaxPage + 10 - pGroup.nMinPage;
                nMax = nCachesize;
                n90pct = nMax * 9 / 10;
                pcache1EnforceMaxPage(pGroup);
                pcache1LeaveMutex(pGroup);
            }
        }

        public int xPagecount()
        {
            pcache1EnterMutex(pGroup);
            var n = (int)nPage;
            pcache1LeaveMutex(pGroup);
            return n;
        }

        public PgHdr xFetch(Pgno key, int createFlag)
        {
            Debug.Assert(bPurgeable || createFlag != 1);
            Debug.Assert(bPurgeable || nMin == 0);
            Debug.Assert(!bPurgeable || nMin == 10);
            Debug.Assert(nMin == 0 || bPurgeable);
            PGroup pGroup;
            pcache1EnterMutex(pGroup = this.pGroup);
            // Step 1: Search the hash table for an existing entry.
            PgHdr1 pPage = null;
            if (nHash > 0)
            {
                var h = (int)(key % nHash);
                for (pPage = apHash[h]; pPage != null && pPage.iKey != key; pPage = pPage.pNext) ;
            }
            // Step 2: Abort if no existing page is found and createFlag is 0
            if (pPage != null || createFlag == 0)
            {
                pcache1PinPage(pPage);
                goto fetch_out;
            }
            // The pGroup local variable will normally be initialized by the pcache1EnterMutex() macro above.  But if SQLITE_MUTEX_OMIT is defined,
            // then pcache1EnterMutex() is a no-op, so we have to initialize the local variable here.  Delaying the initialization of pGroup is an
            // optimization:  The common case is to exit the module before reaching
            // this point.
#if  SQLITE_MUTEX_OMIT
      pGroup = pCache.pGroup;
#endif
            // Step 3: Abort if createFlag is 1 but the cache is nearly full
            var nPinned = nPage - nRecyclable;
            Debug.Assert(nPinned >= 0);
            Debug.Assert(pGroup.mxPinned == pGroup.nMaxPage + 10 - pGroup.nMinPage);
            Debug.Assert(n90pct == nMax * 9 / 10);
            if (createFlag == 1 && (nPinned >= pGroup.mxPinned || nPinned >= (int)n90pct || pcache1UnderMemoryPressure()))
                goto fetch_out;
            if (nPage >= nHash && pcache1ResizeHash() != 0)
                goto fetch_out;
            // Step 4. Try to recycle a page.
            if (bPurgeable && pGroup.pLruTail != null && ((nPage + 1 >= nMax) || pGroup.nCurrentPage >= pGroup.nMaxPage || pcache1UnderMemoryPressure()))
            {
                pPage = pGroup.pLruTail;
                pcache1RemoveFromHash(pPage);
                pcache1PinPage(pPage);
                PCache1 pOtherCache;
                if ((pOtherCache = pPage.pCache).szPage != szPage)
                {
                    pcache1FreePage(ref pPage);
                    pPage = null;
                }
                else
                    pGroup.nCurrentPage -= (pOtherCache.bPurgeable ? 1 : 0) - (bPurgeable ? 1 : 0);
            }
            // Step 5. If a usable page buffer has still not been found, attempt to allocate a new one. 
            if (null == pPage)
            {
                if (createFlag == 1)
                    MallocEx.sqlite3BeginBenignMalloc();
                pcache1LeaveMutex(pGroup);
                pPage = pcache1AllocPage();
                pcache1EnterMutex(pGroup);
                if (createFlag == 1)
                    MallocEx.sqlite3EndBenignMalloc();
            }
            if (pPage != null)
            {
                var h = (int)(key % nHash);
                nPage++;
                pPage.iKey = key;
                pPage.pNext = apHash[h];
                pPage.pCache = this;
                pPage.pLruPrev = null;
                pPage.pLruNext = null;
                PGHDR1_TO_PAGE(pPage).Clear();
                pPage.pPgHdr.pPgHdr1 = pPage;
                apHash[h] = pPage;
            }
        fetch_out:
            if (pPage != null && key > iMaxKey)
                iMaxKey = key;
            pcache1LeaveMutex(pGroup);
            return (pPage != null ? PGHDR1_TO_PAGE(pPage) : null);
        }

        public void xUnpin(PgHdr p2, bool discard)
        {
            var pPage = PAGE_TO_PGHDR1(this, p2);
            Debug.Assert(pPage.pCache == this);
            var pGroup = this.pGroup;
            pcache1EnterMutex(pGroup);
            // It is an error to call this function if the page is already  part of the PGroup LRU list.
            Debug.Assert(pPage.pLruPrev == null && pPage.pLruNext == null);
            Debug.Assert(pGroup.pLruHead != pPage && pGroup.pLruTail != pPage);
            if (discard || pGroup.nCurrentPage > pGroup.nMaxPage)
            {
                pcache1RemoveFromHash(pPage);
                pcache1FreePage(ref pPage);
            }
            else
            {
                // Add the page to the PGroup LRU list. 
                if (pGroup.pLruHead != null)
                {
                    pGroup.pLruHead.pLruPrev = pPage;
                    pPage.pLruNext = pGroup.pLruHead;
                    pGroup.pLruHead = pPage;
                }
                else
                {
                    pGroup.pLruTail = pPage;
                    pGroup.pLruHead = pPage;
                }
                nRecyclable++;
            }
            pcache1LeaveMutex(pGroup);
        }

        public void xRekey(PgHdr p2, Pgno oldKey, Pgno newKey)
        {
            var pPage = PAGE_TO_PGHDR1(this, p2);
            Debug.Assert(pPage.iKey == oldKey);
            Debug.Assert(pPage.pCache == this);
            pcache1EnterMutex(pGroup);
            var h = (int)(oldKey % nHash);
            var pp = apHash[h];
            while (pp != pPage)
                pp = pp.pNext;
            if (pp == apHash[h])
                apHash[h] = pp.pNext;
            else
                pp.pNext = pPage.pNext;
            h = (int)(newKey % nHash);
            pPage.iKey = newKey;
            pPage.pNext = apHash[h];
            apHash[h] = pPage;
            if (newKey > iMaxKey)
                iMaxKey = newKey;
            pcache1LeaveMutex(pGroup);
        }

        public void xTruncate(Pgno iLimit)
        {
            pcache1EnterMutex(pGroup);
            if (iLimit <= iMaxKey)
            {
                pcache1TruncateUnsafe(iLimit);
                iMaxKey = iLimit - 1;
            }
            pcache1LeaveMutex(pGroup);
        }

        public static void xDestroy(ref IPCache pCache)
        {
            PGroup pGroup = pCache.pGroup;
            Debug.Assert(pCache.bPurgeable || (pCache.nMax == 0 && pCache.nMin == 0));
            pcache1EnterMutex(pGroup);
            pCache.pcache1TruncateUnsafe(0);
            pGroup.nMaxPage -= pCache.nMax;
            pGroup.nMinPage -= pCache.nMin;
            pGroup.mxPinned = pGroup.nMaxPage + 10 - pGroup.nMinPage;
            pcache1EnforceMaxPage(pGroup);
            pcache1LeaveMutex(pGroup);
            pCache = null;
        }

#if SQLITE_ENABLE_MEMORY_MANAGEMENT
int sqlite3PcacheReleaseMemory(int nReq){
  int nFree = 0;
  Debug.Assert( sqlite3_mutex_notheld(pcache1.grp.mutex) );
  Debug.Assert( sqlite3_mutex_notheld(pcache1.mutex) );
  if( pcache1.pStart==0 ){
    PgHdr1 p;
    pcache1EnterMutex(&pcache1.grp);
    while( (nReq<0 || nFree<nReq) && ((p=pcache1.grp.pLruTail)!=0) ){
      nFree += pcache1MemSize(PGHDR1_TO_PAGE(p));
      PCache1pinPage(p);
      pcache1RemoveFromHash(p);
      pcache1FreePage(p);
    }
    pcache1LeaveMutex(&pcache1.grp);
  }
  return nFree;
}
#endif
    }
}
