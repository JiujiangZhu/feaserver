using System.Diagnostics;
namespace Contoso.Core
{
    public partial class PCache
    {

#if !DEBUG && SQLITE_ENABLE_EXPENSIVE_ASSERT
        internal int pcacheCheckSynced()
        {
            PgHdr p;
            for (p = pDirtyTail; p != pSynced; p = p.pDirtyPrev)
                Debug.Assert(p.nRef != 0 || (p.flags & PgHdr.PGHDR.NEED_SYNC) != 0);
            return (p == null || p.nRef != 0 || (p.flags & PgHdr.PGHDR.NEED_SYNC) == 0) ? 1 : 0;
        }
#endif

        internal static void pcacheRemoveFromDirtyList(PgHdr pPage)
        {
            var p = pPage.pCache;
            Debug.Assert(pPage.pDirtyNext != null || pPage == p.pDirtyTail);
            Debug.Assert(pPage.pDirtyPrev != null || pPage == p.pDirty);
            // Update the PCache1.pSynced variable if necessary.
            if (p.pSynced == pPage)
            {
                var pSynced = pPage.pDirtyPrev;
                while (pSynced != null && (pSynced.flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                    pSynced = pSynced.pDirtyPrev;
                p.pSynced = pSynced;
            }
            if (pPage.pDirtyNext != null)
                pPage.pDirtyNext.pDirtyPrev = pPage.pDirtyPrev;
            else
            {
                Debug.Assert(pPage == p.pDirtyTail);
                p.pDirtyTail = pPage.pDirtyPrev;
            }
            if (pPage.pDirtyPrev != null)
                pPage.pDirtyPrev.pDirtyNext = pPage.pDirtyNext;
            else
            {
                Debug.Assert(pPage == p.pDirty);
                p.pDirty = pPage.pDirtyNext;
            }
            pPage.pDirtyNext = null;
            pPage.pDirtyPrev = null;
#if SQLITE_ENABLE_EXPENSIVE_ASSERT
            expensive_assert(pcacheCheckSynced(p));
#endif
        }

        internal static void pcacheAddToDirtyList(PgHdr pPage)
        {
            var p = pPage.pCache;
            Debug.Assert(pPage.pDirtyNext == null && pPage.pDirtyPrev == null && p.pDirty != pPage);
            pPage.pDirtyNext = p.pDirty;
            if (pPage.pDirtyNext != null)
            {
                Debug.Assert(pPage.pDirtyNext.pDirtyPrev == null);
                pPage.pDirtyNext.pDirtyPrev = pPage;
            }
            p.pDirty = pPage;
            if (null == p.pDirtyTail)
                p.pDirtyTail = pPage;
            if (null == p.pSynced && 0 == (pPage.flags & PgHdr.PGHDR.NEED_SYNC))
                p.pSynced = pPage;
#if SQLITE_ENABLE_EXPENSIVE_ASSERT
            expensive_assert(pcacheCheckSynced(p));
#endif
        }

        internal static void pcacheUnpin(PgHdr p)
        {
            var pCache = p.pCache;
            if (pCache.bPurgeable)
            {
                if (p.pgno == 1)
                    pCache.pPage1 = null;
                pCache.pCache.xUnpin(p, false);
            }
        }
    }
}
