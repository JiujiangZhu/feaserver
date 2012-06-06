using System;
using System.Diagnostics;
using Contoso.Sys;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public partial class Pager
    {
        const int SQLITE_DEFAULT_JOURNAL_SIZE_LIMIT = -1;
        const int SQLITE_DEFAULT_PAGE_SIZE = 1024;
        const int SQLITE_MAX_DEFAULT_PAGE_SIZE = 8192;
        internal const int SQLITE_MAX_PAGE_SIZE = 65535;
        const int SQLITE_MAX_PAGE_COUNT = 1073741823;
        const int MAX_SECTOR_SIZE = 0x10000;
        const int PAGER_MAX_PGNO = 2147483647;

        public enum PAGER : byte
        {
            OPEN = 0,
            READER = 1,
            WRITER_LOCKED = 2,
            WRITER_CACHEMOD = 3,
            WRITER_DBMOD = 4,
            WRITER_FINISHED = 5,
            ERROR = 6,
        }

        public enum LOCKINGMODE : sbyte
        {
            QUERY = -1,
            NORMAL = 0,
            EXCLUSIVE = 1,
        }

        [Flags]
        public enum JOURNALMODE : sbyte
        {
            QUERY = -1,
            DELETE = 0,
            PERSIST = 1,
            OFF = 2,
            TRUNCATE = 3,
            MEMORY = 4,
            WAL = 5,
        }

        [Flags]
        public enum PAGEROPEN : byte
        {
            OMIT_JOURNAL = 0x0001,
            NO_READLOCK = 0x0002,
            MEMORY = 0x0004,
        }

        public VirtualFileSystem pVfs;            // OS functions to use for IO
        public bool exclusiveMode;          // Boolean. True if locking_mode==EXCLUSIVE
        public JOURNALMODE journalMode;     // One of the PAGER_JOURNALMODE_* values
        public byte useJournal;             // Use a rollback journal on this file
        public byte noReadlock;             // Do not bother to obtain readlocks
        public bool noSync;                 // Do not sync the journal if true
        public bool fullSync;               // Do extra syncs of the journal for robustness
        public VirtualFile.SYNC ckptSyncFlags;          // SYNC_NORMAL or SYNC_FULL for checkpoint
        public VirtualFile.SYNC syncFlags;              // SYNC_NORMAL or SYNC_FULL otherwise
        public bool tempFile;               // zFilename is a temporary file
        public bool readOnly;               // True for a read-only database
        public bool alwaysRollback;         // Disable DontRollback() for all pages
        public byte memDb;                  // True to inhibit all file I/O
        // The following block contains those class members that change during routine opertion.  Class members not in this block are either fixed
        // when the pager is first created or else only change when there is a significant mode change (such as changing the page_size, locking_mode,
        // or the journal_mode).  From another view, these class members describe the "state" of the pager, while other class members describe the
        // "configuration" of the pager.
        public PAGER eState;                // Pager state (OPEN, READER, WRITER_LOCKED..) 
        public VirtualFile.LOCK eLock;      // Current lock held on database file 
        public bool changeCountDone;        // Set after incrementing the change-counter 
        public int setMaster;               // True if a m-j name has been written to jrnl 
        public byte doNotSpill;             // Do not spill the cache when non-zero 
        public byte doNotSyncSpill;         // Do not do a spill that requires jrnl sync 
        public byte subjInMemory;           // True to use in-memory sub-journals 
        public Pgno dbSize;                 // Number of pages in the database 
        public Pgno dbOrigSize;             // dbSize before the current transaction 
        public Pgno dbFileSize;             // Number of pages in the database file 
        public Pgno dbHintSize;             // Value passed to FCNTL_SIZE_HINT call 
        public SQLITE errCode;              // One of several kinds of errors 
        public int nRec;                    // Pages journalled since last j-header written 
        public uint cksumInit;              // Quasi-random value added to every checksum 
        public uint nSubRec;                // Number of records written to sub-journal 
        public Bitvec pInJournal;           // One bit for each page in the database file 
        public VirtualFile fd;             // File descriptor for database 
        public VirtualFile jfd;            // File descriptor for main journal 
        public VirtualFile sjfd;           // File descriptor for sub-journal 
        public long journalOff;             // Current write offset in the journal file 
        public long journalHdr;             // Byte offset to previous journal header 
        public IBackup pBackup;             // Pointer to list of ongoing backup processes 
        public PagerSavepoint[] aSavepoint; // Array of active savepoints 
        public int nSavepoint;              // Number of elements in aSavepoint[] 
        public byte[] dbFileVers = new byte[16];    // Changes whenever database file changes
        // End of the routinely-changing class members
        public ushort nExtra;               // Add this many bytes to each in-memory page
        public short nReserve;              // Number of unused bytes at end of each page
        public VirtualFileSystem.OPEN vfsFlags;               // Flags for VirtualFileSystem.xOpen() 
        public uint sectorSize;             // Assumed sector size during rollback 
        public int pageSize;                // Number of bytes in a page 
        public Pgno mxPgno;                 // Maximum allowed size of the database 
        public long journalSizeLimit;       // Size limit for persistent journal files 
        public string zFilename;            // Name of the database file 
        public string zJournal;             // Name of the journal file 
        public Func<object, int> xBusyHandler;  // Function to call when busy 
        public object pBusyHandlerArg;      // Context argument for xBusyHandler 
#if DEBUG
        public int nHit, nMiss;             // Cache hits and missing 
        public int nRead, nWrite;           // Database pages read/written 
#else
        public int nHit;
#endif
        public Action<PgHdr> xReiniter;     // Call this routine when reloading pages
#if SQLITE_HAS_CODEC
        public codec_ctx.dxCodec xCodec;                 // Routine for en/decoding data
        public codec_ctx.dxCodecSizeChng xCodecSizeChng; // Notify of page size changes
        public codec_ctx.dxCodecFree xCodecFree;         // Destructor for the codec
        public codec_ctx pCodec;               // First argument to xCodec... methods
#endif
        public byte[] pTmpSpace;               // Pager.pageSize bytes of space for tmp use
        public PCache pPCache;                 // Pointer to page cache object
#if !SQLITE_OMIT_WAL
        public Wal pWal;                       // Write-ahead log used by "journal_mode=wal"
        public string zWal;                    // File name for write-ahead log
#else
        public Wal pWal = null;             // Having this dummy here makes C# easier
#endif

        internal static Pgno PAGER_MJ_PGNO(Pager x) { return ((Pgno)((VirtualFile.PENDING_BYTE / ((x).pageSize)) + 1)); }

#if SQLITE_HAS_CODEC
        internal static bool CODEC1(Pager P, byte[] D, uint N, int X) { return (P.xCodec != null && P.xCodec(P.pCodec, D, N, X) == null); }
        internal static bool CODEC2(Pager P, byte[] D, uint N, int X, ref byte[] O)
        {
            if (P.xCodec == null) { O = D;/*do nothing*/ return false; }
            else return ((O = P.xCodec(P.pCodec, D, N, X)) == null);
        }
#else
        internal static bool CODEC1(Pager P, byte[] D, uint N, int X) { return false; }
        internal static bool CODEC2(Pager P, byte[] D, uint N, int X, ref byte[] O) { O = D; return false; }
#endif

#if false && !SQLITE_OMIT_WAL
        internal static int pagerUseWal(Pager pPager) { return (pPager.pWal != 0); }
#else
        internal bool pagerUseWal() { return false; }
        internal SQLITE pagerRollbackWal() { return SQLITE.OK; }
        internal SQLITE pagerWalFrames(PgHdr w, Pgno x, int y, VirtualFile.SYNC z) { return SQLITE.OK; }
        internal SQLITE pagerOpenWalIfPresent() { return SQLITE.OK; }
        internal SQLITE pagerBeginReadTransaction() { return SQLITE.OK; }
#endif

#if SQLITE_ENABLE_ATOMIC_WRITE
        internal static int jrnlBufferSize(Pager pPager)
        {
            Debug.Assert(0 == MEMDB);
            if (!pPager.tempFile)
            {
                Debug.Assert(pPager.fd.isOpen);
                var dc = sqlite3OsDeviceCharacteristics(pPager.fd);
                var nSector = pPager.sectorSize;
                var szPage = pPager.pageSize;
                Debug.Assert(SQLITE_IOCAP_ATOMIC512 == (512 >> 8));
                Debug.Assert(SQLITE_IOCAP_ATOMIC64K == (65536 >> 8));
                if (0 == (dc & (SQLITE_IOCAP_ATOMIC | (szPage >> 8)) || nSector > szPage))
                    return 0;
            }
            return JOURNAL_HDR_SZ(pPager) + JOURNAL_PG_SZ(pPager);
        }
#endif

#if !SQLITE_CHECK_PAGES
        internal static uint pager_pagehash(PgHdr pPage) { return pager_datahash(pPage.pPager.pageSize, pPage.pData); }
        internal static uint pager_datahash(int nByte, byte[] pData)
        {
            uint hash = 0;
            for (var i = 0; i < nByte; i++)
                hash = (hash * 1039) + pData[i];
            return hash;
        }
        internal static void pager_set_pagehash(PgHdr pPage) { pPage.pageHash = pager_pagehash(pPage); }
        internal static void checkPage(PgHdr pPg)
        {
            var pPager = pPg.pPager;
            Debug.Assert(pPager.eState != PAGER.ERROR);
            Debug.Assert((pPg.flags & PgHdr.PGHDR.DIRTY) != 0 || pPg.pageHash == pager_pagehash(pPg));
        }
#else
        internal static uint pager_pagehash(PgHdr X) { return 0; }
        internal static uint pager_datahash(int X, byte[] Y) { return 0; }
        internal static void pager_set_pagehash(PgHdr X) { }
#endif
    }
}
