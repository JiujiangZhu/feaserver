using System;
using Contoso.Sys;
using IPCache = Contoso.Core.Name.PCache1;
namespace Contoso.Core
{
    public partial class PCache
    {
        public PgHdr pDirty;        // List of dirty pages in LRU order
        public PgHdr pDirtyTail;    
        public PgHdr pSynced;       // Last synced page in dirty page list
        public int nRef;            // Number of referenced pages
        public int nMax;            // Configured cache size
        public int szPage;          // Size of every page in this cache
        public int szExtra;         // Size of extra space for each page
        public bool bPurgeable;     // True if pages are on backing store
        public Func<object, PgHdr, SQLITE> xStress;   // Call to try make a page clean
        public object pStress;      // Argument to xStress
        public IPCache pCache;      // Pluggable cache module
        public PgHdr pPage1;        // Reference to page 1

        public void Clear()
        {
            pDirty = null;
            pDirtyTail = null;
            pSynced = null;
            nRef = 0;
        }
    }
}
