using System;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public class PgHdr : DbPage
    {
        public enum PGHDR : int
        {
            DIRTY = 0x002, // Page has changed
            NEED_SYNC = 0x004, // Fsync the rollback journal before writing this page to the database
            NEED_READ = 0x008, // Content is unread
            REUSE_UNLIKELY = 0x010, // A hint that reuse is unlikely
            DONT_WRITE = 0x020, // Do not write content to disk
        }

        public byte[] pData;          /* Content of this page */
        public MemPage pExtra;        /* Extra content */
        public PgHdr pDirty;          /* Transient list of dirty pages */
        public Pgno pgno;             /* The page number for this page */
        public Pager pPager;          /* The pager to which this page belongs */
#if DEBUG
        public uint pageHash;          /* Hash of page content */
#endif
        public PGHDR flags;             /* PGHDR flags defined below */

        // Elements above are public.  All that follows is private to pcache.c and should not be accessed by other modules.

        public int nRef;              /* Number of users of this page */
        public PCache pCache;         /* Cache that owns this page */
        public bool CacheAllocated;   /* True, if allocated from cache */

        public PgHdr pDirtyNext;      /* Next element in list of dirty pages */
        public PgHdr pDirtyPrev;      /* Previous element in list of dirty pages */
        public PgHdr1 pPgHdr1;        /* Cache page header this this page */

        public static implicit operator bool(PgHdr b)
        {
            return (b != null);
        }

        public void Clear()
        {
            this.pData = null;
            this.pExtra = null;
            this.pDirty = null;
            this.pgno = 0;
            this.pPager = null;
#if DEBUG
            this.pageHash = 0;
#endif
            this.flags = 0;
            this.nRef = 0;
            this.CacheAllocated = false;
            this.pCache = null;
            this.pDirtyNext = null;
            this.pDirtyPrev = null;
            this.pPgHdr1 = null;
        }
    }
}
