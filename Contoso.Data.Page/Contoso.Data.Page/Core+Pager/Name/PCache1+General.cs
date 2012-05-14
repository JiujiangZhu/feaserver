using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using System.Text;
namespace Contoso.Core.Name
{
    public partial class PCache1
    {
        internal SQLITE pcache1ResizeHash()
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pGroup.mutex));
            var nNew = nHash * 2;
            if (nNew < 256)
                nNew = 256;
            pcache1LeaveMutex(pGroup);
            if (nHash != 0)
                MallocEx.sqlite3BeginBenignMalloc();
            var apNew = new PgHdr1[nNew];
            if (nHash != 0)
                MallocEx.sqlite3EndBenignMalloc();
            pcache1EnterMutex(pGroup);
            if (apNew != null)
            {
                for (var i = 0; i < nHash; i++)
                {
                    var pNext = apHash[i];
                    PgHdr1 pPage;
                    while ((pPage = pNext) != null)
                    {
                        var h = (Pgno)(pPage.iKey % nNew);
                        pNext = pPage.pNext;
                        pPage.pNext = apNew[h];
                        apNew[h] = pPage;
                    }
                }
                apHash = apNew;
                nHash = nNew;
            }
            return (apHash != null ? SQLITE.OK : SQLITE.NOMEM);
        }

        internal static void pcache1PinPage(PgHdr1 pPage)
        {
            if (pPage == null)
                return;
            var pCache = pPage.pCache;
            var pGroup = pCache.pGroup;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pGroup.mutex));
            if (pPage.pLruNext != null || pPage == pGroup.pLruTail)
            {
                if (pPage.pLruPrev != null)
                    pPage.pLruPrev.pLruNext = pPage.pLruNext;
                if (pPage.pLruNext != null)
                    pPage.pLruNext.pLruPrev = pPage.pLruPrev;
                if (pGroup.pLruHead == pPage)
                    pGroup.pLruHead = pPage.pLruNext;
                if (pGroup.pLruTail == pPage)
                    pGroup.pLruTail = pPage.pLruPrev;
                pPage.pLruNext = null;
                pPage.pLruPrev = null;
                pPage.pCache.nRecyclable--;
            }
        }

        internal static void pcache1RemoveFromHash(PgHdr1 pPage)
        {
            var pCache = pPage.pCache;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pCache.pGroup.mutex));
            var h = (int)(pPage.iKey % pCache.nHash);
            PgHdr1 pPrev = null;
            PgHdr1 pp;
            for (pp = pCache.apHash[h]; pp != pPage; pPrev = pp, pp = pp.pNext) ;
            if (pPrev == null)
                pCache.apHash[h] = pp.pNext;
            else
                pPrev.pNext = pp.pNext;
            pCache.nPage--;
        }

        internal static void pcache1EnforceMaxPage(PGroup pGroup)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pGroup.mutex));
            while (pGroup.nCurrentPage > pGroup.nMaxPage && pGroup.pLruTail != null)
            {
                PgHdr1 p = pGroup.pLruTail;
                Debug.Assert(p.pCache.pGroup == pGroup);
                pcache1PinPage(p);
                pcache1RemoveFromHash(p);
                pcache1FreePage(ref p);
            }
        }

        internal void pcache1TruncateUnsafe(uint iLimit)
        {
#if !DEBUG
            uint nPage = 0;
#endif
            Debug.Assert(MutexEx.sqlite3_mutex_held(pGroup.mutex));
            for (uint h = 0; h < nHash; h++)
            {
                var pp = apHash[h];
                PgHdr1 pPrev = null;
                PgHdr1 pPage;
                while ((pPage = pp) != null)
                {
                    if (pPage.iKey >= iLimit)
                    {
                        nPage--;
                        pp = pPage.pNext;
                        pcache1PinPage(pPage);
                        if (apHash[h] == pPage)
                            apHash[h] = pPage.pNext;
                        else
                            pPrev.pNext = pp;
                        pcache1FreePage(ref pPage);
                    }
                    else
                    {
                        pp = pPage.pNext;
#if !DEBUG
                        nPage++;
#endif
                    }
                    pPrev = pPage;
                }
            }
#if !DEBUG
            Debug.Assert(pCache.nPage == nPage);
#endif
        }
    }
}
