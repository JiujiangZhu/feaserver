using Pgno = System.UInt32;
using System;
namespace Contoso.Core
{
    public partial class Btree
    {
        public enum TRANS : byte
        {
            NONE = 0,
            READ = 1,
            WRITE = 2,
        }

        [Flags]
        public enum OPEN
        {
            OMIT_JOURNAL = 1, // Do not create or use a rollback journal 
            NO_READLOCK = 2,  // Omit readlocks on readonly files 
            MEMORY = 4,       // This is an in-memory DB 
            SINGLE = 8,       // The file contains at most 1 b-tree 
            UNORDERED = 16,   // Use of a hash implementation is OK 
        }

        public sqlite3 db;        // The database connection holding this Btree 
        public BtShared pBt;      // Sharable content of this Btree 
        public TRANS inTrans;     // TRANS_NONE, TRANS_READ or TRANS_WRITE 
        public bool sharable;     // True if we can share pBt with another db 
        public bool locked;       // True if db currently has pBt locked 
        public int wantToLock;    // Number of nested calls to sqlite3BtreeEnter() 
        public int nBackup;       // Number of backup operations reading this btree 
        public Btree pNext;       // List of other sharable Btrees from the same db 
        public Btree pPrev;       // Back pointer of the same list 
#if !SQLITE_OMIT_SHARED_CACHE
        BtLock lock;              // Object used to lock page 1 
#endif

        //static Pgno PENDING_BYTE_PAGE(BtShared pBt) { return Pager.PAGER_MJ_PGNO(pBt.pPager); }
        //static Pgno PTRMAP_PAGENO(BtShared pBt, Pgno pgno) { return ptrmapPageno(pBt, pgno); }
        //static Pgno PTRMAP_PTROFFSET(Pgno pgptrmap, Pgno pgno) { return (5 * (pgno - pgptrmap - 1)); }
        //static bool PTRMAP_ISPAGE(BtShared pBt, Pgno pgno) { return (PTRMAP_PAGENO((pBt), (pgno)) == (pgno)); }
    }
}
