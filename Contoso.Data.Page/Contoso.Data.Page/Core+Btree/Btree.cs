﻿using System;
using System.Diagnostics;
using System.Text;
using Mem = System.Object;

namespace Contoso.Core
{
    public partial class Btree
    {
        const int SQLITE_DEFAULT_CACHE_SIZE = 2000;
        const string SQLITE_FILE_HEADER = "SQLite format 3\0";
        internal static byte[] zMagicHeader = Encoding.UTF8.GetBytes(SQLITE_FILE_HEADER);
        const int MASTER_ROOT = 1;
        const int EXTRA_SIZE = 0;
        internal static IVdbe _vdbe;

        public enum TRANS : byte
        {
            NONE = 0,
            READ = 1,
            WRITE = 2,
        }

        [Flags] // was:BTREE_*
        public enum OPEN : byte
        {
            OMIT_JOURNAL = 1, // Do not create or use a rollback journal 
            NO_READLOCK = 2,  // Omit readlocks on readonly files 
            MEMORY = 4,       // This is an in-memory DB 
            SINGLE = 8,       // The file contains at most 1 b-tree 
            UNORDERED = 16,   // Use of a hash implementation is OK 
        }

        // was:BTREE_*
        public enum CREATETABLE
        {
            INTKEY = 1,
            BLOBKEY = 2,
        }

        // was:BTREE_*
        public enum META : byte
        {
            FREE_PAGE_COUNT = 0,
            SCHEMA_VERSION = 1,
            FILE_FORMAT = 2,
            DEFAULT_CACHE_SIZE = 3,
            LARGEST_ROOT_PAGE = 4,
            TEXT_ENCODING = 5,
            USER_VERSION = 6,
            INCR_VACUUM = 7,
        }

        // was:BTREE_AUTOVACUUM_*
        public enum AUTOVACUUM
        {
            DEFAULT = NONE,
            NONE = 0,        // Do not do auto-vacuum
            FULL = 1,        // Do full auto-vacuum
            INCR = 2,        // Incremental vacuum
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


        // HOOKS
#if !SQLITE_OMIT_SHARED_CACHE
//void sqlite3BtreeEnter(Btree);
//void sqlite3BtreeEnterAll(sqlite3);
#else
        internal void sqlite3BtreeEnter() { }
        internal void sqlite3BtreeLeave() { }
#endif
#if !SQLITE_OMIT_SHARED_CACHE && SQLITE_THREADSAFE
//int sqlite3BtreeSharable(Btree);
//void sqlite3BtreeLeave(Btree);
//void sqlite3BtreeEnterCursor(BtCursor);
//void sqlite3BtreeLeaveCursor(BtCursor);
//void sqlite3BtreeLeaveAll(sqlite3);
#if !NDEBUG
// These routines are used inside Debug.Assert() statements only.
int sqlite3BtreeHoldsMutex(Btree);
int sqlite3BtreeHoldsAllMutexes(sqlite3);
int sqlite3SchemaMutexHeld(sqlite3*,int,Schema);
#endif
#else
        //internal bool sqlite3BtreeSharable() { return false; }
        //internal void sqlite3BtreeLeave() { }
        //internal static void sqlite3BtreeEnterCursor(BtCursor X) { }
        //internal static void sqlite3BtreeLeaveCursor(BtCursor X) { }
        //internal static void sqlite3BtreeLeaveAll(sqlite3 X) { }
        internal bool sqlite3BtreeHoldsMutex() { return true; }
        //internal static bool sqlite3BtreeHoldsAllMutexes(sqlite3 X) { return true; }
        //internal static bool sqlite3SchemaMutexHeld(sqlite3 X, int y, ISchema z) { return true; }
#endif
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
