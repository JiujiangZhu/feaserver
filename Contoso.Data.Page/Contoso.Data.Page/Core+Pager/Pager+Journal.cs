using System;
using Pgno = System.UInt32;
using Contoso.Sys;
using System.Diagnostics;
using System.Text;
using LOCK = Contoso.Sys.VirtualFile.LOCK;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal static readonly byte[] aJournalMagic = new byte[] { 0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7 };
        internal static uint JOURNAL_PG_SZ(Pager pPager) { return (uint)pPager.pageSize + 8; }
        internal static uint JOURNAL_HDR_SZ(Pager pPager) { return pPager.sectorSize; }

#if SQLITE_ENABLE_ATOMIC_WRITE
//  int sqlite3JournalOpen(sqlite3_vfs *, string , sqlite3_file *, int, int);
//  int sqlite3JournalSize(sqlite3_vfs );
//  int sqlite3JournalCreate(sqlite3_file );
#else
        internal static int sqlite3JournalSize(VirtualFileSystem pVfs) { return pVfs.szOsFile; }
#endif

        internal static bool subjRequiresPage(PgHdr pPg)
        {
            var pgno = pPg.pgno;
            var pPager = pPg.pPager;
            for (var i = 0; i < pPager.nSavepoint; i++)
            {
                var p = pPager.aSavepoint[i];
                if (p.nOrig >= pgno && p.pInSavepoint.sqlite3BitvecTest(pgno) == 0)
                    return true;
            }
            return false;
        }

        internal static bool pageInJournal(PgHdr pPg)
        {
            return pPg.pPager.pInJournal.sqlite3BitvecTest(pPg.pgno) != 0;
        }

        internal static SQLITE readMasterJournal(VirtualFile pJrnl, byte[] zMaster, uint nMaster)
        {
            int len = 0;                // Length in bytes of master journal name 
            long szJ = 0;               // Total size in bytes of journal file pJrnl 
            uint cksum = 0;             // MJ checksum value read from journal
            zMaster[0] = 0;
            var aMagic = new byte[8];   // A buffer to hold the magic header
            SQLITE rc;
            if (SQLITE.OK != (rc = FileEx.sqlite3OsFileSize(pJrnl, ref szJ))
                || szJ < 16
                || SQLITE.OK != (rc = read32bits(pJrnl, (int)(szJ - 16), ref len))
                || len >= nMaster
                || SQLITE.OK != (rc = read32bits(pJrnl, szJ - 12, ref cksum))
                || SQLITE.OK != (rc = FileEx.sqlite3OsRead(pJrnl, aMagic, 8, szJ - 8))
                || ArrayEx.Compare(aMagic, aJournalMagic, 8) != 0
                || SQLITE.OK != (rc = FileEx.sqlite3OsRead(pJrnl, zMaster, len, (long)(szJ - 16 - len))))
                return rc;
            // See if the checksum matches the master journal name
            for (var u = 0; u < len; u++)
                cksum -= zMaster[u];
            if (cksum != 0)
                // If the checksum doesn't add up, then one or more of the disk sectors containing the master journal filename is corrupted. This means
                // definitely roll back, so just return SQLITE.OK and report a (nul) master-journal filename.
                len = 0;
            if (len == 0)
                zMaster[0] = 0;
            return SQLITE.OK;
        }

        internal static long journalHdrOffset(Pager pPager)
        {
            long offset = 0;
            var c = pPager.journalOff;
            if (c != 0)
                offset = (int)(((c - 1) / pPager.sectorSize + 1) * pPager.sectorSize);
            Debug.Assert(offset % pPager.sectorSize == 0);
            Debug.Assert(offset >= c);
            Debug.Assert((offset - c) < pPager.sectorSize);
            return offset;
        }

        internal static void seekJournalHdr(Pager pPager)
        {
            pPager.journalOff = journalHdrOffset(pPager);
        }

        internal static SQLITE zeroJournalHdr(Pager pPager, int doTruncate)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.jfd.isOpen);
            if (pPager.journalOff != 0)
            {
                var iLimit = pPager.journalSizeLimit; // Local cache of jsl
                SysEx.IOTRACE("JZEROHDR %p\n", pPager);
                if (doTruncate != 0 || iLimit == 0)
                    rc = FileEx.sqlite3OsTruncate(pPager.jfd, 0);
                else
                {
                    var zeroHdr = new byte[28];
                    rc = FileEx.sqlite3OsWrite(pPager.jfd, zeroHdr, zeroHdr.Length, 0);
                }
                if (rc == SQLITE.OK && !pPager.noSync)
                    rc = FileEx.sqlite3OsSync(pPager.jfd, VirtualFile.SYNC.DATAONLY | pPager.syncFlags);
                // At this point the transaction is committed but the write lock is still held on the file. If there is a size limit configured for
                // the persistent journal and the journal file currently consumes more space than that limit allows for, truncate it now. There is no need
                // to sync the file following this operation.
                if (rc == SQLITE.OK && iLimit > 0)
                {
                    long sz = 0;
                    rc = FileEx.sqlite3OsFileSize(pPager.jfd, ref sz);
                    if (rc == SQLITE.OK && sz > iLimit)
                        rc = FileEx.sqlite3OsTruncate(pPager.jfd, iLimit);
                }
            }
            return rc;
        }

        internal static SQLITE writeJournalHdr(Pager pPager)
        {
            Debug.Assert(pPager.jfd.isOpen);    // Journal file must be open.
            var nHeader = (uint)pPager.pageSize; // Size of buffer pointed to by zHeader
            if (nHeader > JOURNAL_HDR_SZ(pPager))
                nHeader = JOURNAL_HDR_SZ(pPager);
            // If there are active savepoints and any of them were created since the most recent journal header was written, update the
            // PagerSavepoint.iHdrOffset fields now.
            for (var ii = 0; ii < pPager.nSavepoint; ii++)
                if (pPager.aSavepoint[ii].iHdrOffset == 0)
                    pPager.aSavepoint[ii].iHdrOffset = pPager.journalOff;
            pPager.journalHdr = pPager.journalOff = journalHdrOffset(pPager);
            // Write the nRec Field - the number of page records that follow this journal header. Normally, zero is written to this value at this time.
            // After the records are added to the journal (and the journal synced, if in full-sync mode), the zero is overwritten with the true number
            // of records (see syncJournal()).
            // A faster alternative is to write 0xFFFFFFFF to the nRec field. When reading the journal this value tells SQLite to assume that the
            // rest of the journal file contains valid page records. This assumption is dangerous, as if a failure occurred whilst writing to the journal
            // file it may contain some garbage data. There are two scenarios where this risk can be ignored:
            //   * When the pager is in no-sync mode. Corruption can follow a power failure in this case anyway.
            //   * When the SQLITE_IOCAP_SAFE_APPEND flag is set. This guarantees that garbage data is never appended to the journal file.
            Debug.Assert(pPager.fd.isOpen || pPager.noSync);
            var zHeader = pPager.pTmpSpace;  // Temporary space used to build header
            if (pPager.noSync || (pPager.journalMode == JOURNALMODE.MEMORY) || (FileEx.sqlite3OsDeviceCharacteristics(pPager.fd) & VirtualFile.IOCAP.SAFE_APPEND) != 0)
            {
                aJournalMagic.CopyTo(zHeader, 0);
                ConvertEx.put32bits(zHeader, aJournalMagic.Length, 0xffffffff);
            }
            else
                Array.Clear(zHeader, 0, aJournalMagic.Length + 4);
            // The random check-hash initialiser
            long i64Temp = 0;
            UtilEx.sqlite3_randomness(sizeof(long), ref i64Temp);
            pPager.cksumInit = (Pgno)i64Temp;
            ConvertEx.put32bits(zHeader, aJournalMagic.Length + 4, pPager.cksumInit);
            // The initial database size
            ConvertEx.put32bits(zHeader, aJournalMagic.Length + 8, pPager.dbOrigSize);
            // The assumed sector size for this process
            ConvertEx.put32bits(zHeader, aJournalMagic.Length + 12, pPager.sectorSize);
            // The page size
            ConvertEx.put32bits(zHeader, aJournalMagic.Length + 16, (uint)pPager.pageSize);
            // Initializing the tail of the buffer is not necessary.  Everything works find if the following memset() is omitted.  But initializing
            // the memory prevents valgrind from complaining, so we are willing to take the performance hit.
            Array.Clear(zHeader, aJournalMagic.Length + 20, (int)nHeader - aJournalMagic.Length + 20);
            // In theory, it is only necessary to write the 28 bytes that the journal header consumes to the journal file here. Then increment the
            // Pager.journalOff variable by JOURNAL_HDR_SZ so that the next record is written to the following sector (leaving a gap in the file
            // that will be implicitly filled in by the OS).
            // However it has been discovered that on some systems this pattern can be significantly slower than contiguously writing data to the file,
            // even if that means explicitly writing data to the block of (JOURNAL_HDR_SZ - 28) bytes that will not be used. So that is what
            // is done.
            // The loop is required here in case the sector-size is larger than the database page size. Since the zHeader buffer is only Pager.pageSize
            // bytes in size, more than one call to sqlite3OsWrite() may be required to populate the entire journal header sector.
            SQLITE rc = SQLITE.OK;                 // Return code
            for (uint nWrite = 0; rc == SQLITE.OK && nWrite < JOURNAL_HDR_SZ(pPager); nWrite += nHeader)
            {
                SysEx.IOTRACE("JHDR %p %lld %d\n", pPager, pPager.journalHdr, nHeader);
                rc = FileEx.sqlite3OsWrite(pPager.jfd, zHeader, (int)nHeader, pPager.journalOff);
                Debug.Assert(pPager.journalHdr <= pPager.journalOff);
                pPager.journalOff += (int)nHeader;
            }
            return rc;
        }

        internal static SQLITE readJournalHdr(Pager pPager, int isHot, long journalSize, out uint pNRec, out Pgno pDbSize)
        {
            var aMagic = new byte[8]; // A buffer to hold the magic header
            Debug.Assert(pPager.jfd.isOpen);    // Journal file must be open.
            pNRec = 0;
            pDbSize = 0;
            // Advance Pager.journalOff to the start of the next sector. If the journal file is too small for there to be a header stored at this
            // point, return SQLITE_DONE.
            pPager.journalOff = journalHdrOffset(pPager);
            if (pPager.journalOff + JOURNAL_HDR_SZ(pPager) > journalSize)
                return SQLITE.DONE;
            var iHdrOff = pPager.journalOff; // Offset of journal header being read
            // Read in the first 8 bytes of the journal header. If they do not match the magic string found at the start of each journal header, return
            // SQLITE_DONE. If an IO error occurs, return an error code. Otherwise, proceed.
            SQLITE rc;
            if (isHot != 0 || iHdrOff != pPager.journalHdr)
            {
                rc = FileEx.sqlite3OsRead(pPager.jfd, aMagic, aMagic.Length, iHdrOff);
                if (rc != SQLITE.OK)
                    return rc;
                if (ArrayEx.Compare(aMagic, aJournalMagic, aMagic.Length) != 0)
                    return SQLITE.DONE;
            }
            // Read the first three 32-bit fields of the journal header: The nRec field, the checksum-initializer and the database size at the start
            // of the transaction. Return an error code if anything goes wrong.
            if (SQLITE.OK != (rc = read32bits(pPager.jfd, iHdrOff + 8, ref pNRec))
                || SQLITE.OK != (rc = read32bits(pPager.jfd, iHdrOff + 12, ref pPager.cksumInit))
                || SQLITE.OK != (rc = read32bits(pPager.jfd, iHdrOff + 16, ref pDbSize)))
                return rc;
            if (pPager.journalOff == 0)
            {
                // Read the page-size and sector-size journal header fields.
                uint iPageSize = 0;     // Page-size field of journal header
                uint iSectorSize = 0;   // Sector-size field of journal header
                if (SQLITE.OK != (rc = read32bits(pPager.jfd, iHdrOff + 20, ref iSectorSize))
                    || SQLITE.OK != (rc = read32bits(pPager.jfd, iHdrOff + 24, ref iPageSize)))
                    return rc;
                // Versions of SQLite prior to 3.5.8 set the page-size field of the journal header to zero. In this case, assume that the Pager.pageSize
                // variable is already set to the correct page size.
                if (iPageSize == 0)
                    iPageSize = (uint)pPager.pageSize;
                // Check that the values read from the page-size and sector-size fields are within range. To be 'in range', both values need to be a power
                // of two greater than or equal to 512 or 32, and not greater than their respective compile time maximum limits.
                if (iPageSize < 512 || iSectorSize < 32
                    || iPageSize > SQLITE_MAX_PAGE_SIZE || iSectorSize > MAX_SECTOR_SIZE
                    || ((iPageSize - 1) & iPageSize) != 0 || ((iSectorSize - 1) & iSectorSize) != 0)
                    // If the either the page-size or sector-size in the journal-header is invalid, then the process that wrote the journal-header must have
                    // crashed before the header was synced. In this case stop reading the journal file here.
                    return SQLITE.DONE;
                // Update the page-size to match the value read from the journal. Use a testcase() macro to make sure that malloc failure within
                // PagerSetPagesize() is tested.
                rc = sqlite3PagerSetPagesize(pPager, ref iPageSize, -1);
                // Update the assumed sector-size to match the value used by the process that created this journal. If this journal was
                // created by a process other than this one, then this routine is being called from within pager_playback(). The local value
                // of Pager.sectorSize is restored at the end of that routine.
                pPager.sectorSize = iSectorSize;
            }
            pPager.journalOff += (int)JOURNAL_HDR_SZ(pPager);
            return rc;
        }

        internal static SQLITE writeMasterJournal(Pager pPager, string zMaster)
        {
            Debug.Assert(pPager.setMaster == 0);
            Debug.Assert(!pagerUseWal(pPager));
            if (null == zMaster
                || pPager.journalMode == JOURNALMODE.MEMORY
                || pPager.journalMode == JOURNALMODE.OFF)
                return SQLITE.OK;
            pPager.setMaster = 1;
            Debug.Assert(pPager.jfd.isOpen);
            Debug.Assert(pPager.journalHdr <= pPager.journalOff);
            // Calculate the length in bytes and the checksum of zMaster
            uint cksum = 0; // Checksum of string zMaster
            int nMaster;    // Length of string zMaster
            for (nMaster = 0; nMaster < zMaster.Length && zMaster[nMaster] != 0; nMaster++)
                cksum += zMaster[nMaster];
            // If in full-sync mode, advance to the next disk sector before writing the master journal name. This is in case the previous page written to
            // the journal has already been synced.
            if (pPager.fullSync)
                pPager.journalOff = journalHdrOffset(pPager);
            var iHdrOff = pPager.journalOff; // Offset of header in journal file
            // Write the master journal data to the end of the journal file. If an error occurs, return the error code to the caller.
            SQLITE rc;
            if (SQLITE.OK != (rc = write32bits(pPager.jfd, iHdrOff, (uint)PAGER_MJ_PGNO(pPager)))
                || SQLITE.OK != (rc = FileEx.sqlite3OsWrite(pPager.jfd, Encoding.UTF8.GetBytes(zMaster), nMaster, iHdrOff + 4))
                || SQLITE.OK != (rc = write32bits(pPager.jfd, iHdrOff + 4 + nMaster, (uint)nMaster))
                || SQLITE.OK != (rc = write32bits(pPager.jfd, iHdrOff + 4 + nMaster + 4, cksum))
                || SQLITE.OK != (rc = FileEx.sqlite3OsWrite(pPager.jfd, aJournalMagic, 8, iHdrOff + 4 + nMaster + 8)))
                return rc;
            pPager.journalOff += nMaster + 20;
            // If the pager is in peristent-journal mode, then the physical journal-file may extend past the end of the master-journal name
            // and 8 bytes of magic data just written to the file. This is dangerous because the code to rollback a hot-journal file
            // will not be able to find the master-journal name to determine whether or not the journal is hot.
            // Easiest thing to do in this scenario is to truncate the journal file to the required size.
            long jrnlSize = 0;  // Size of journal file on disk
            if (SQLITE.OK == (rc = FileEx.sqlite3OsFileSize(pPager.jfd, ref jrnlSize)) && jrnlSize > pPager.journalOff)
                rc = FileEx.sqlite3OsTruncate(pPager.jfd, pPager.journalOff);
            return rc;
        }

        internal static SQLITE pagerSyncHotJournal(Pager pPager)
        {
            var rc = SQLITE.OK;
            if (!pPager.noSync)
                rc = FileEx.sqlite3OsSync(pPager.jfd, VirtualFile.SYNC.NORMAL);
            if (rc == SQLITE.OK)
                rc = FileEx.sqlite3OsFileSize(pPager.jfd, ref pPager.journalHdr);
            return rc;
        }

        internal static SQLITE syncJournal(Pager pPager, int newHdr)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.eState == PAGER.WRITER_CACHEMOD || pPager.eState == PAGER.WRITER_DBMOD);
            Debug.Assert(assert_pager_state(pPager));
            Debug.Assert(!pagerUseWal(pPager));
            rc = sqlite3PagerExclusiveLock(pPager);
            if (rc != SQLITE.OK)
                return rc;
            if (!pPager.noSync)
            {
                Debug.Assert(!pPager.tempFile);
                if (pPager.jfd.isOpen && pPager.journalMode != JOURNALMODE.MEMORY)
                {
                    var iDc = FileEx.sqlite3OsDeviceCharacteristics(pPager.fd);
                    Debug.Assert(pPager.jfd.isOpen);
                    if (0 == (iDc & VirtualFile.IOCAP.SAFE_APPEND))
                    {
                        // This block deals with an obscure problem. If the last connection that wrote to this database was operating in persistent-journal
                        // mode, then the journal file may at this point actually be larger than Pager.journalOff bytes. If the next thing in the journal
                        // file happens to be a journal-header (written as part of the previous connection's transaction), and a crash or power-failure
                        // occurs after nRec is updated but before this connection writes anything else to the journal file (or commits/rolls back its
                        // transaction), then SQLite may become confused when doing the hot-journal rollback following recovery. It may roll back all
                        // of this connections data, then proceed to rolling back the old, out-of-date data that follows it. Database corruption.
                        // To work around this, if the journal file does appear to contain a valid header following Pager.journalOff, then write a 0x00
                        // byte to the start of it to prevent it from being recognized.
                        // Variable iNextHdrOffset is set to the offset at which this problematic header will occur, if it exists. aMagic is used
                        // as a temporary buffer to inspect the first couple of bytes of the potential journal header.
                        var zHeader = new byte[aJournalMagic.Length + 4];
                        aJournalMagic.CopyTo(zHeader, 0);
                        ConvertEx.put32bits(zHeader, aJournalMagic.Length, pPager.nRec);
                        var iNextHdrOffset = journalHdrOffset(pPager);
                        var aMagic = new byte[8];
                        rc = FileEx.sqlite3OsRead(pPager.jfd, aMagic, 8, iNextHdrOffset);
                        if (rc == SQLITE.OK && 0 == ArrayEx.Compare(aMagic, aJournalMagic, 8))
                        {
                            var zerobyte = new byte[1];
                            rc = FileEx.sqlite3OsWrite(pPager.jfd, zerobyte, 1, iNextHdrOffset);
                        }
                        if (rc != SQLITE.OK && rc != SQLITE.IOERR_SHORT_READ)
                            return rc;
                        // Write the nRec value into the journal file header. If in full-synchronous mode, sync the journal first. This ensures that
                        // all data has really hit the disk before nRec is updated to mark it as a candidate for rollback.
                        // This is not required if the persistent media supports the SAFE_APPEND property. Because in this case it is not possible
                        // for garbage data to be appended to the file, the nRec field is populated with 0xFFFFFFFF when the journal header is written
                        // and never needs to be updated.
                        if (pPager.fullSync && 0 == (iDc & VirtualFile.IOCAP.SEQUENTIAL))
                        {
                            PAGERTRACE("SYNC journal of %d\n", PAGERID(pPager));
                            SysEx.IOTRACE("JSYNC %p\n", pPager);
                            rc = FileEx.sqlite3OsSync(pPager.jfd, pPager.syncFlags);
                            if (rc != SQLITE.OK)
                                return rc;
                        }
                        SysEx.IOTRACE("JHDR %p %lld\n", pPager, pPager.journalHdr);
                        rc = FileEx.sqlite3OsWrite(pPager.jfd, zHeader, zHeader.Length, pPager.journalHdr);
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                    if (0 == (iDc & VirtualFile.IOCAP.SEQUENTIAL))
                    {
                        PAGERTRACE("SYNC journal of %d\n", PAGERID(pPager));
                        SysEx.IOTRACE("JSYNC %p\n", pPager);
                        rc = FileEx.sqlite3OsSync(pPager.jfd, pPager.syncFlags | (pPager.syncFlags == VirtualFile.SYNC.FULL ? VirtualFile.SYNC.DATAONLY : 0));
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                    pPager.journalHdr = pPager.journalOff;
                    if (newHdr != 0 && 0 == (iDc & VirtualFile.IOCAP.SAFE_APPEND))
                    {
                        pPager.nRec = 0;
                        rc = writeJournalHdr(pPager);
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                }
                else
                    pPager.journalHdr = pPager.journalOff;
            }
            // Unless the pager is in noSync mode, the journal file was just successfully synced. Either way, clear the PGHDR_NEED_SYNC flag on all pages.
            pPager.pPCache.sqlite3PcacheClearSyncFlags();
            pPager.eState = PAGER.WRITER_DBMOD;
            Debug.Assert(assert_pager_state(pPager));
            return SQLITE.OK;
        }

        internal static SQLITE openSubJournal(Pager pPager)
        {
            var rc = SQLITE.OK;
            if (!pPager.sjfd.isOpen)
            {
                if (pPager.journalMode == JOURNALMODE.MEMORY || pPager.subjInMemory != 0)
                    pPager.sjfd = new MemJournalFile();
                else
                    rc = pagerOpentemp(pPager, ref pPager.sjfd, VirtualFileSystem.OPEN.SUBJOURNAL);
            }
            return rc;
        }

        internal static SQLITE subjournalPage(PgHdr pPg)
        {
            var rc = SQLITE.OK;
            var pPager = pPg.pPager;
            if (pPager.journalMode != JOURNALMODE.OFF)
            {
                // Open the sub-journal, if it has not already been opened
                Debug.Assert(pPager.useJournal != 0);
                Debug.Assert(pPager.jfd.isOpen || pPager.pagerUseWal());
                Debug.Assert(pPager.sjfd.isOpen || pPager.nSubRec == 0);
                Debug.Assert(pPager.pagerUseWal() || pageInJournal(pPg) || pPg.pgno > pPager.dbOrigSize);
                rc = openSubJournal(pPager);
                // If the sub-journal was opened successfully (or was already open), write the journal record into the file. 
                if (rc == SQLITE.OK)
                {
                    var pData = pPg.pData;
                    long offset = pPager.nSubRec * (4 + pPager.pageSize);
                    byte[] pData2 = null;
                    if (CODEC2(pPager, pData, pPg.pgno, codec_ctx.ENCRYPT_READ_CTX, ref pData2))
                        return SQLITE.NOMEM;
                    PAGERTRACE("STMT-JOURNAL %d page %d\n", PAGERID(pPager), pPg.pgno);
                    rc = write32bits(pPager.sjfd, offset, pPg.pgno);
                    if (rc == SQLITE.OK)
                        rc = FileEx.sqlite3OsWrite(pPager.sjfd, pData2, pPager.pageSize, offset + 4);
                }
            }
            if (rc == SQLITE.OK)
            {
                pPager.nSubRec++;
                Debug.Assert(pPager.nSavepoint > 0);
                rc = addToSavepointBitvecs(pPager, pPg.pgno);
            }
            return rc;
        }

        internal static SQLITE hasHotJournal(Pager pPager, ref int pExists)
        {
            var pVfs = pPager.pVfs;
            var exists = 1;               // True if a journal file is present
            var jrnlOpen = (pPager.jfd.isOpen ? 1 : 0);
            Debug.Assert(pPager.useJournal != 0);
            Debug.Assert(pPager.fd.isOpen);
            Debug.Assert(pPager.eState == PAGER.OPEN);
            Debug.Assert(jrnlOpen == 0 || (FileEx.sqlite3OsDeviceCharacteristics(pPager.jfd) & VirtualFile.IOCAP.UNDELETABLE_WHEN_OPEN) != 0);
            pExists = 0;
            var rc = SQLITE.OK;
            if (0 == jrnlOpen)
                rc = FileEx.sqlite3OsAccess(pVfs, pPager.zJournal, VirtualFileSystem.ACCESS.EXISTS, ref exists);
            if (rc == SQLITE.OK && exists != 0)
            {
                int locked = 0;                 // True if some process holds a RESERVED lock
                // Race condition here:  Another process might have been holding the the RESERVED lock and have a journal open at the sqlite3OsAccess()
                // call above, but then delete the journal and drop the lock before we get to the following sqlite3OsCheckReservedLock() call.  If that
                // is the case, this routine might think there is a hot journal when in fact there is none.  This results in a false-positive which will
                // be dealt with by the playback routine.
                rc = FileEx.sqlite3OsCheckReservedLock(pPager.fd, ref locked);
                if (rc == SQLITE.OK && locked == 0)
                {
                    Pgno nPage = 0; // Number of pages in database file
                    // Check the size of the database file. If it consists of 0 pages, then delete the journal file. See the header comment above for
                    // the reasoning here.  Delete the obsolete journal file under a RESERVED lock to avoid race conditions and to avoid violating [H33020].
                    rc = pagerPagecount(pPager, ref nPage);
                    if (rc == SQLITE.OK)
                    {
                        if (nPage == 0)
                        {
                            MallocEx.sqlite3BeginBenignMalloc();
                            if (pagerLockDb(pPager, LOCK.RESERVED) == SQLITE.OK)
                            {
                                FileEx.sqlite3OsDelete(pVfs, pPager.zJournal, 0);
                                if (!pPager.exclusiveMode)
                                    pagerUnlockDb(pPager, LOCK.SHARED);
                            }
                            MallocEx.sqlite3EndBenignMalloc();
                        }
                        else
                        {
                            // The journal file exists and no other connection has a reserved or greater lock on the database file. Now check that there is
                            // at least one non-zero bytes at the start of the journal file. If there is, then we consider this journal to be hot. If not,
                            // it can be ignored.
                            if (0 == jrnlOpen)
                            {
                                var f = VirtualFileSystem.OPEN.READONLY | VirtualFileSystem.OPEN.MAIN_JOURNAL;
                                rc = FileEx.sqlite3OsOpen(pVfs, pPager.zJournal, pPager.jfd, f, ref f);
                            }
                            if (rc == SQLITE.OK)
                            {
                                var first = new byte[1];
                                rc = FileEx.sqlite3OsRead(pPager.jfd, first, 1, 0);
                                if (rc == SQLITE.IOERR_SHORT_READ)
                                    rc = SQLITE.OK;
                                if (0 == jrnlOpen)
                                    FileEx.sqlite3OsClose(pPager.jfd);
                                pExists = (first[0] != 0) ? 1 : 0;
                            }
                            else if (rc == SQLITE.CANTOPEN)
                            {
                                // If we cannot open the rollback journal file in order to see if its has a zero header, that might be due to an I/O error, or
                                // it might be due to the race condition.  Either way, assume that the journal is hot. This might be a false positive.  But if it is, then the
                                // automatic journal playback and recovery mechanism will deal with it under an EXCLUSIVE lock where we do not need to
                                // worry so much with race conditions.
                                pExists = 1;
                                rc = SQLITE.OK;
                            }
                        }
                    }
                }
            }
            return rc;
        }
    }
}
