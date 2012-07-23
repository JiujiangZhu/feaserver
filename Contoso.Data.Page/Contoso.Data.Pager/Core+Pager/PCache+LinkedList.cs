using System.Diagnostics;
namespace Contoso.Core
{
    public partial class PCache
    {

#if !DEBUG && SQLITE_ENABLE_EXPENSIVE_ASSERT
        private int pcacheCheckSynced()
        {
            PgHdr p;
            for (p = pDirtyTail; p != pSynced; p = p.pDirtyPrev)
                Debug.Assert(p.nRef != 0 || (p.flags & PgHdr.PGHDR.NEED_SYNC) != 0);
            return (p == null || p.nRef != 0 || (p.flags & PgHdr.PGHDR.NEED_SYNC) == 0) ? 1 : 0;
        }
#endif

        private static void pcacheRemoveFromDirtyList(PgHdr pPage)
        {
            var p = pPage.Cache;
            Debug.Assert(pPage.DirtyNext != null || pPage == p.pDirtyTail);
            Debug.Assert(pPage.DirtyPrev != null || pPage == p.pDirty);
            // Update the PCache1.pSynced variable if necessary.
            if (p.pSynced == pPage)
            {
                var pSynced = pPage.DirtyPrev;
                while (pSynced != null && (pSynced.Flags & PgHdr.PGHDR.NEED_SYNC) != 0)
                    pSynced = pSynced.DirtyPrev;
                p.pSynced = pSynced;
            }
            if (pPage.DirtyNext != null)
                pPage.DirtyNext.DirtyPrev = pPage.DirtyPrev;
            else
            {
                Debug.Assert(pPage == p.pDirtyTail);
                p.pDirtyTail = pPage.DirtyPrev;
            }
            if (pPage.DirtyPrev != null)
                pPage.DirtyPrev.DirtyNext = pPage.DirtyNext;
            else
            {
                Debug.Assert(pPage == p.pDirty);
                p.pDirty = pPage.DirtyNext;
            }
            pPage.DirtyNext = null;
            pPage.DirtyPrev = null;
#if SQLITE_ENABLE_EXPENSIVE_ASSERT
            expensive_assert(pcacheCheckSynced(p));
#endif
        }

        private static void pcacheAddToDirtyList(PgHdr pPage)
        {
            var p = pPage.Cache;
            Debug.Assert(pPage.DirtyNext == null && pPage.DirtyPrev == null && p.pDirty != pPage);
            pPage.DirtyNext = p.pDirty;
            if (pPage.DirtyNext != null)
            {
                Debug.Assert(pPage.DirtyNext.DirtyPrev == null);
                pPage.DirtyNext.DirtyPrev = pPage;
            }
            p.pDirty = pPage;
            if (null == p.pDirtyTail)
                p.pDirtyTail = pPage;
            if (null == p.pSynced && 0 == (pPage.Flags & PgHdr.PGHDR.NEED_SYNC))
                p.pSynced = pPage;
#if SQLITE_ENABLE_EXPENSIVE_ASSERT
            expensive_assert(pcacheCheckSynced(p));
#endif
        }

        private static void pcacheUnpin(PgHdr p)
        {
            var pCache = p.Cache;
            if (pCache.bPurgeable)
            {
                if (p.ID == 1)
                    pCache.pPage1 = null;
                pCache.pCache.xUnpin(p, false);
            }
        }
    }
}
