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
using System;
namespace FeaServer.Engine.Core
{
    public class Pager
    {
        public sqlite3_vfs pVfs;           /* OS functions to use for IO */
        public bool exclusiveMode;         /* Boolean. True if locking_mode==EXCLUSIVE */
        public byte journalMode;             /* One of the PAGER_JOURNALMODE_* values */
        public byte useJournal;              /* Use a rollback journal on this file */
        public byte noReadlock;              /* Do not bother to obtain readlocks */
        public bool noSync;                /* Do not sync the journal if true */
        public bool fullSync;              /* Do extra syncs of the journal for robustness */
        public byte ckptSyncFlags;           /* SYNC_NORMAL or SYNC_FULL for checkpoint */
        public byte syncFlags;               /* SYNC_NORMAL or SYNC_FULL otherwise */
        public bool tempFile;              /* zFilename is a temporary file */
        public bool readOnly;              /* True for a read-only database */
        public bool alwaysRollback;        /* Disable DontRollback() for all pages */
        public byte memDb;                   /* True to inhibit all file I/O */
        /**************************************************************************
        ** The following block contains those class members that change during
        ** routine opertion.  Class members not in this block are either fixed
        ** when the pager is first created or else only change when there is a
        ** significant mode change (such as changing the page_size, locking_mode,
        ** or the journal_mode).  From another view, these class members describe
        ** the "state" of the pager, while other class members describe the
        ** "configuration" of the pager.
        */
        public byte eState;                  /* Pager state (OPEN, READER, WRITER_LOCKED..) */
        public byte eLock;                   /* Current lock held on database file */
        public bool changeCountDone;       /* Set after incrementing the change-counter */
        public int setMaster;              /* True if a m-j name has been written to jrnl */
        public byte doNotSpill;              /* Do not spill the cache when non-zero */
        public byte doNotSyncSpill;          /* Do not do a spill that requires jrnl sync */
        public byte subjInMemory;            /* True to use in-memory sub-journals */
        public uint dbSize;                /* Number of pages in the database */
        public uint dbOrigSize;            /* dbSize before the current transaction */
        public uint dbFileSize;            /* Number of pages in the database file */
        public uint dbHintSize;            /* Value passed to FCNTL_SIZE_HINT call */
        public int errCode;                /* One of several kinds of errors */
        public int nRec;                   /* Pages journalled since last j-header written */
        public uint cksumInit;              /* Quasi-random value added to every checksum */
        public uint nSubRec;                /* Number of records written to sub-journal */
        public Bitvec pInJournal;          /* One bit for each page in the database file */
        public sqlite3_file fd;            /* File descriptor for database */
        public sqlite3_file jfd;           /* File descriptor for main journal */
        public sqlite3_file sjfd;          /* File descriptor for sub-journal */
        public long journalOff;             /* Current write offset in the journal file */
        public long journalHdr;             /* Byte offset to previous journal header */
        public sqlite3_backup pBackup;     /* Pointer to list of ongoing backup processes */
        public PagerSavepoint[] aSavepoint;/* Array of active savepoints */
        public int nSavepoint;             /* Number of elements in aSavepoint[] */
        public byte[] dbFileVers = new byte[16];/* Changes whenever database file changes */
        /*
        ** End of the routinely-changing class members
        ***************************************************************************/

        public ushort nExtra;                 /* Add this many bytes to each in-memory page */
        public short nReserve;               /* Number of unused bytes at end of each page */
        public uint vfsFlags;               /* Flags for sqlite3_vfs.xOpen() */
        public uint sectorSize;             /* Assumed sector size during rollback */
        public int pageSize;               /* Number of bytes in a page */
        public uint mxPgno;                /* Maximum allowed size of the database */
        public long journalSizeLimit;       /* Size limit for persistent journal files */
        public string zFilename;           /* Name of the database file */
        public string zJournal;            /* Name of the journal file */
        public dxBusyHandler xBusyHandler; /* Function to call when busy */
        public object pBusyHandlerArg;     /* Context argument for xBusyHandler */
#if DEBUG
        public int nHit, nMiss;              /* Cache hits and missing */
        public int nRead, nWrite;            /* Database pages read/written */
#else
        public int nHit;
#endif
        public dxReiniter xReiniter; //(DbPage*,int);/* Call this routine when reloading pages */
#if SQLITE_HAS_CODEC
        public dxCodec xCodec;                 /* Routine for en/decoding data */
        public dxCodecSizeChng xCodecSizeChng; /* Notify of page size changes */
        public dxCodecFree xCodecFree;         /* Destructor for the codec */
        public codec_ctx pCodec;               /* First argument to xCodec... methods */
#endif
        public byte[] pTmpSpace;               /* Pager.pageSize bytes of space for tmp use */
        public PCache pPCache;                 /* Pointer to page cache object */
#if !SQLITE_OMIT_WAL
        public Wal pWal;                       /* Write-ahead log used by "journal_mode=wal" */
        public string zWal;                    /* File name for write-ahead log */
#else
        public sqlite3_vfs pWal = null;             /* Having this dummy here makes C# easier */
#endif
    }
}
