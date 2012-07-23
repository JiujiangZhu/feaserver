using System;
using Pgno = System.UInt32;
using Contoso.Threading;
using Contoso.Collections;

namespace Contoso.Core
{
    public partial class BtShared
    {
        private static int _versions;
        internal int Version;
        public BtShared() { Version = _versions++; }

        public Pager Pager;             // The page cache
        public sqlite3b DB;              // Database connection currently using this Btree
        public BtreeCursor Cursors;     // A list of all open cursors
        public MemPage Page1;           // First page of the database
        public bool ReadOnly;           // True if the underlying file is readonly
        public bool PageSizeFixed;      // True if the page size can no longer be changed
        public bool SecureDelete;       // True if secure_delete is enabled
        public bool InitiallyEmpty;     // Database is empty at start of transaction
        public Btree.OPEN OpenFlags;    // Flags to sqlite3BtreeOpen()
#if !SQLITE_OMIT_AUTOVACUUM
        public bool AutoVacuum;         // True if auto-vacuum is enabled
        public bool IncrVacuum;         // True if incr-vacuum is enabled
#endif
        public Btree.TRANS InTransaction;   // Transaction state
        public bool DoNotUseWal;        // If true, do not open write-ahead-log file
        public ushort MaxLocal;         // Maximum local payload in non-LEAFDATA tables
        public ushort MinLocal;         // Minimum local payload in non-LEAFDATA tables
        public ushort MaxLeaf;          // Maximum local payload in a LEAFDATA table
        public ushort MinLeaf;          // Minimum local payload in a LEAFDATA table
        public uint PageSize;           // Total number of bytes on a page
        public uint UsableSize;         // Number of usable bytes on each page
        public int Transactions;        // Number of open transactions (read + write)
        public Pgno Pages;              // Number of pages in the database
        public ISchema Schema;          // Pointer to space allocated by sqlite3BtreeSchema()
        public Action<ISchema> xFreeSchema; // Destructor for BtShared.pSchema
        public sqlite3_mutex Mutex;     // Non-recursive mutex required to access this object
        public Bitvec HasContent;       // Set of pages moved to free-list this transaction
#if !SQLITE_OMIT_SHARED_CACHE
        public int nRef;                // Number of references to this structure
        public BtShared Next;           // Next on a list of sharable BtShared structs
        public Btree.BtreeLock Locks;   // List of locks held on this shared-btree struct
        public Btree Writer;            // Btree with currently open write transaction
        public bool IsExclusive;        // True if pWriter has an EXCLUSIVE lock on the db
        public bool IsPending;          // If waiting for read-locks to clear
#endif
        public byte[] pTmpSpace;        // BtShared.pageSize bytes of space for tmp use
    }
}
