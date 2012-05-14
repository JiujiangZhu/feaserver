using Pgno = System.UInt32;
namespace Contoso.Core.Name
{
    public class PgHdr1
    {
        public Pgno iKey;                   // Key value (page number)
        public PgHdr1 pNext;                // Next in hash table chain
        public PCache1 pCache;              // Cache that currently owns this page
        public PgHdr1 pLruNext;             // Next in LRU list of unpinned pages
        public PgHdr1 pLruPrev;             // Previous in LRU list of unpinned pages

        public PgHdr pPgHdr = new PgHdr();  // Pointer to Actual Page Header

        public void Clear()
        {
            this.iKey = 0;
            this.pNext = null;
            this.pCache = null;
            this.pPgHdr.Clear();
        }
    }
}
