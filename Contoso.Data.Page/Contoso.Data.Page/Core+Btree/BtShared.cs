using Contoso.Sys;
using System;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public class BtShared
    {
        public Pager pPager;            // The page cache
        public sqlite3 db;              // Database connection currently using this Btree
        public BtCursor pCursor;        // A list of all open cursors
        public MemPage pPage1;          // First page of the database
        public bool readOnly;           // True if the underlying file is readonly
        public bool pageSizeFixed;      // True if the page size can no longer be changed
        public bool secureDelete;       // True if secure_delete is enabled
        public bool initiallyEmpty;     // Database is empty at start of transaction
        public byte openFlags;          // Flags to sqlite3BtreeOpen()
#if !SQLITE_OMIT_AUTOVACUUM
        public bool autoVacuum;         // True if auto-vacuum is enabled
        public bool incrVacuum;         // True if incr-vacuum is enabled
#endif
        public Btree.TRANS inTransaction;      // Transaction state
        public bool doNotUseWAL;        // If true, do not open write-ahead-log file
        public ushort maxLocal;         // Maximum local payload in non-LEAFDATA tables
        public ushort minLocal;         // Minimum local payload in non-LEAFDATA tables
        public ushort maxLeaf;          // Maximum local payload in a LEAFDATA table
        public ushort minLeaf;          // Minimum local payload in a LEAFDATA table
        public uint pageSize;           // Total number of bytes on a page
        public uint usableSize;         // Number of usable bytes on each page
        public int nTransaction;        // Number of open transactions (read + write)
        public Pgno nPage;              // Number of pages in the database
        public ISchema pSchema;          // Pointer to space allocated by sqlite3BtreeSchema()
        public Action<ISchema> xFreeSchema;// Destructor for BtShared.pSchema
        public sqlite3_mutex mutex;     // Non-recursive mutex required to access this object
        public Bitvec pHasContent;      // Set of pages moved to free-list this transaction
#if !SQLITE_OMIT_SHARED_CACHE
        public int nRef;                // Number of references to this structure
        public BtShared pNext;          // Next on a list of sharable BtShared structs
        public BtLock pLock;            // List of locks held on this shared-btree struct
        public Btree pWriter;           // Btree with currently open write transaction
        public byte isExclusive;        // True if pWriter has an EXCLUSIVE lock on the db
        public byte isPending;          // If waiting for read-locks to clear
#endif
        public byte[] pTmpSpace;        // BtShared.pageSize bytes of space for tmp use
    }
}
