using Pgno = System.UInt32;
using IPgHdr = Contoso.Core.Name.PgHdr1;
using IPCache = Contoso.Core.Name.PCache1;
using Contoso.Threading;
namespace Contoso.Core.Name
{
    public partial class PCache1
    {
        // Cache configuration parameters. Page size (szPage) and the purgeable flag (bPurgeable) are set when the cache is created. nMax may be 
        // modified at any time by a call to the pcache1CacheSize() method. The PGroup mutex must be held when accessing nMax.
        public PGroup pGroup;           // PGroup this cache belongs to
        public int szPage;              // Size of allocated pages in bytes
        public bool bPurgeable;         // True if cache is purgeable
        public int nMin;                // Minimum number of pages reserved
        public int nMax;                // Configured "cache_size" value
        public int n90pct;              // nMax*9/10
        // Hash table of all pages. The following variables may only be accessed when the accessor is holding the PGroup mutex.
        public int nRecyclable;         // Number of pages in the LRU list
        public int nPage;               // Total number of pages in apHash
        public int nHash;               // Number of slots in apHash[]
        public PgHdr1[] apHash;         // Hash table for fast lookup by key
        //
        public Pgno iMaxKey;            // Largest key seen since xTruncate()

        public void Clear()
        {
            nRecyclable = 0;
            nPage = 0;
            nHash = 0;
            apHash = null;
            iMaxKey = 0;
        }

        private static PgHdr PGHDR1_TO_PAGE(PgHdr1 p) { return p.pPgHdr; }
        private static PgHdr1 PAGE_TO_PGHDR1(PCache1 c, PgHdr p) { return (PgHdr1)p.PgHdr1; }
        private static void pcache1EnterMutex(PGroup X) { MutexEx.sqlite3_mutex_enter(X.mutex); }
        private static void pcache1LeaveMutex(PGroup X) { MutexEx.sqlite3_mutex_leave(X.mutex); }
    }
}
