using IPgHdr = Contoso.Core.Name.PgHdr1;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public class PgHdr
    {
        public enum PGHDR : int
        {
            DIRTY = 0x002, // Page has changed
            NEED_SYNC = 0x004, // Fsync the rollback journal before writing this page to the database
            NEED_READ = 0x008, // Content is unread
            REUSE_UNLIKELY = 0x010, // A hint that reuse is unlikely
            DONT_WRITE = 0x020, // Do not write content to disk
        }

        public byte[] Data;          // Content of this page
        public object Extra;        // Extra content
        public PgHdr Dirtys;          // Transient list of dirty pages
        public Pgno ID;             // The page number for this page
        public Pager Pager;          // The pager to which this page belongs
#if DEBUG
        public uint PageHash;          // Hash of page content
#endif
        public PGHDR Flags;             // PGHDR flags defined below

        // Elements above are public.  All that follows is private to pcache.c and should not be accessed by other modules.

        public int Refs;              // Number of users of this page
        public PCache Cache;         // Cache that owns this page
        public bool CacheAllocated;   // True, if allocated from cache

        public PgHdr DirtyNext;      // Next element in list of dirty pages
        public PgHdr DirtyPrev;      // Previous element in list of dirty pages
        public IPgHdr PgHdr1;        // Cache page header this this page

        public static implicit operator bool(PgHdr b) { return (b != null); }

        public void ClearState()
        {
            this.Data = null;
            this.Extra = null;
            this.Dirtys = null;
            this.ID = 0;
            this.Pager = null;
#if DEBUG
            this.PageHash = 0;
#endif
            this.Flags = 0;
            this.Refs = 0;
            this.CacheAllocated = false;
            this.Cache = null;
            this.DirtyNext = null;
            this.DirtyPrev = null;
            this.PgHdr1 = null;
        }
    }
}
