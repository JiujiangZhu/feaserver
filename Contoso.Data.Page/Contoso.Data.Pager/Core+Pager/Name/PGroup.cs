using Contoso.Threading;
namespace Contoso.Core.Name
{
    public class PGroup
    {
        public sqlite3_mutex mutex = new sqlite3_mutex(); // MUTEX_STATIC_LRU or NULL
        public int nMaxPage;                  // Sum of nMax for purgeable caches
        public int nMinPage;                  // Sum of nMin for purgeable caches
        public int mxPinned;                  // nMaxpage + 10 - nMinPage
        public int nCurrentPage;              // Number of purgeable pages allocated
        public PgHdr1 pLruHead, pLruTail;     // LRU list of unpinned pages
    }
}
