using Mem = System.Object;
using Pgno = System.UInt32;
using System;
using System.Diagnostics;

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

        public class UnpackedRecord
        {
            [Flags]
            public enum UNPACKED : ushort
            {
                NEED_FREE = 0x0001,     // Memory is from sqlite3Malloc()
                NEED_DESTROY = 0x0002,  // apMem[]s should all be destroyed
                IGNORE_ROWID = 0x0004,  // Ignore trailing rowid on key1
                INCRKEY = 0x0008,       // Make this key an epsilon larger
                PREFIX_MATCH = 0x0010,  // A prefix match is considered OK
                PREFIX_SEARCH = 0x0020, // A prefix match is considered OK
            }

            public KeyInfo pKeyInfo;    // Collation and sort-order information
            public ushort nField;       // Number of entries in apMem[]
            public UNPACKED flags;      // Boolean settings.  UNPACKED_... below
            public long rowid;          // Used by UNPACKED_PREFIX_SEARCH
            public Mem[] aMem;          // Values
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

        internal static int MX_CELL_SIZE(BtShared pBt) { return (int)(pBt.pageSize - 8); }
        internal static int MX_CELL(BtShared pBt) { return ((int)(pBt.pageSize - 8) / 6); }

        internal const byte PTF_INTKEY = 0x01;
        internal const byte PTF_ZERODATA = 0x02;
        internal const byte PTF_LEAFDATA = 0x04;
        internal const byte PTF_LEAF = 0x08;


        internal void sqlite3BtreeEnter() { }
        internal void sqlite3BtreeLeave() { }
#if DEBUG
        internal void btreeIntegrity()
        {
            Debug.Assert(pBt.inTransaction != TRANS.NONE || pBt.nTransaction == 0);
            Debug.Assert(pBt.inTransaction >= inTrans);
        }
#else
        internal void btreeIntegrity() { }
#endif

    }
}
