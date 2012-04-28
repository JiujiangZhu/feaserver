#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
using FeaServer.Engine.System;
namespace FeaServer.Engine.Core
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
        public byte inTransaction;      // Transaction state
        public bool doNotUseWAL;        // If true, do not open write-ahead-log file
        public ushort maxLocal;         // Maximum local payload in non-LEAFDATA tables
        public ushort minLocal;         // Minimum local payload in non-LEAFDATA tables
        public ushort maxLeaf;          // Maximum local payload in a LEAFDATA table
        public ushort minLeaf;          // Minimum local payload in a LEAFDATA table
        public uint pageSize;           // Total number of bytes on a page
        public uint usableSize;         // Number of usable bytes on each page
        public int nTransaction;        // Number of open transactions (read + write)
        public uint nPage;              // Number of pages in the database
        public Schema pSchema;          // Pointer to space allocated by sqlite3BtreeSchema()
        public dxFreeSchema xFreeSchema;// Destructor for BtShared.pSchema
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
