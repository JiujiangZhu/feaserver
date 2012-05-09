using System;
using Pgno = System.UInt32;
using Contoso.Sys;
using System.Diagnostics;
using System.Text;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal static readonly byte[] aJournalMagic = new byte[] { 0xd9, 0xd5, 0x05, 0xf9, 0x20, 0xa1, 0x63, 0xd7 };
        internal static uint JOURNAL_PG_SZ(Pager pPager) { return (uint)pPager.pageSize + 8; }
        internal static uint JOURNAL_HDR_SZ(Pager pPager) { return pPager.sectorSize; }

        // Return true if it is necessary to write page *pPg into the sub-journal.  A page needs to be written into the sub-journal if there exists one
        // or more open savepoints for which:
        //   * The page-number is less than or equal to PagerSavepoint.nOrig, and
        //   * The bit corresponding to the page-number is not set in PagerSavepoint.pInSavepoint.
        internal static bool subjRequiresPage(PgHdr pPg)
        {
            var pgno = pPg.pgno;
            var pPager = pPg.pPager;
            for (var i = 0; i < pPager.nSavepoint; i++)
            {
                var p = pPager.aSavepoint[i];
                if (p.nOrig >= pgno && sqlite3BitvecTest(p.pInSavepoint, pgno) == 0)
                    return true;
            }
            return false;
        }

        // Return true if the page is already in the journal file.
        internal static bool pageInJournal(PgHdr pPg)
        {
            return sqlite3BitvecTest(pPg.pPager.pInJournal, pPg.pgno) != 0;
        }

        // When this is called the journal file for pager pPager must be open. This function attempts to read a master journal file name from the
        // end of the file and, if successful, copies it into memory supplied by the caller. See comments above writeMasterJournal() for the format
        // used to store a master journal file name at the end of a journal file.
        //
        // zMaster must point to a buffer of at least nMaster bytes allocated by the caller. This should be sqlite3_vfs.mxPathname+1 (to ensure there is
        // enough space to write the master journal name). If the master journal name in the journal is longer than nMaster bytes (including a
        // nul-terminator), then this is handled as if no master journal name were present in the journal.
        //
        // If a master journal file name is present at the end of the journal file, then it is copied into the buffer pointed to by zMaster. A
        // nul-terminator byte is appended to the buffer following the master journal file name.
        //
        // If it is determined that no master journal file name is present zMaster[0] is set to 0 and SQLITE_OK returned.
        //
        // If an error occurs while reading from the journal file, an SQLite error code is returned.
        internal static SQLITE readMasterJournal(sqlite3_file pJrnl, byte[] zMaster, uint nMaster)
        {
            int len = 0;                // Length in bytes of master journal name 
            long szJ = 0;               // Total size in bytes of journal file pJrnl 
            uint cksum = 0;             // MJ checksum value read from journal
            zMaster[0] = 0;
            var aMagic = new byte[8];   // A buffer to hold the magic header
            SQLITE rc;
            if (SQLITE.OK != (rc = sqlite3OsFileSize(pJrnl, ref szJ))
                || szJ < 16
                || SQLITE.OK != (rc = read32bits(pJrnl, (int)(szJ - 16), ref len))
                || len >= nMaster
                || SQLITE.OK != (rc = read32bits(pJrnl, szJ - 12, ref cksum))
                || SQLITE.OK != (rc = sqlite3OsRead(pJrnl, aMagic, 8, szJ - 8))
                || ArrayEx.Compare(aMagic, aJournalMagic, 8) != 0
                || SQLITE.OK != (rc = sqlite3OsRead(pJrnl, zMaster, len, (long)(szJ - 16 - len))))
                return rc;
            // See if the checksum matches the master journal name
            for (var u = 0; u < len; u++)
                cksum -= zMaster[u];
            if (cksum != 0)
                // If the checksum doesn't add up, then one or more of the disk sectors containing the master journal filename is corrupted. This means
                // definitely roll back, so just return SQLITE_OK and report a (nul) master-journal filename.
                len = 0;
            if (len == 0)
                zMaster[0] = 0;
            return SQLITE.OK;
        }

        // Return the offset of the sector boundary at or immediately following the value in pPager.journalOff, assuming a sector
        // size of pPager.sectorSize bytes.
        //
        // i.e for a sector size of 512:
        //
        //   Pager.journalOff          Return value
        //   ---------------------------------------
        //   0                         0
        //   512                       512
        //   100                       512
        //   2000                      2048
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

        // The journal file must be open when this function is called.
        // This function is a no-op if the journal file has not been written to within the current transaction (i.e. if Pager.journalOff==0).
        // If doTruncate is non-zero or the Pager.journalSizeLimit variable is set to 0, then truncate the journal file to zero bytes in size. Otherwise,
        // zero the 28-byte header at the start of the journal file. In either case, if the pager is not in no-sync mode, sync the journal file immediately
        // after writing or truncating it.
        // If Pager.journalSizeLimit is set to a positive, non-zero value, and following the truncation or zeroing described above the size of the
        // journal file in bytes is larger than this value, then truncate the journal file to Pager.journalSizeLimit bytes. The journal file does
        // not need to be synced following this operation.
        // If an IO error occurs, abandon processing and return the IO error code. Otherwise, return SQLITE_OK.
        internal static SQLITE zeroJournalHdr(Pager pPager, int doTruncate)
        {
            var rc = SQLITE.OK;
            Debug.Assert(pPager.jfd.isOpen);
            if (pPager.journalOff != 0)
            {
                var iLimit = pPager.journalSizeLimit; // Local cache of jsl
                IOTRACE("JZEROHDR %p\n", pPager);
                if (doTruncate != 0 || iLimit == 0)
                    rc = sqlite3OsTruncate(pPager.jfd, 0);
                else
                {
                    var zeroHdr = new byte[28];
                    rc = sqlite3OsWrite(pPager.jfd, zeroHdr, zeroHdr.Length, 0);
                }
                if (rc == SQLITE.OK && !pPager.noSync)
                    rc = sqlite3OsSync(pPager.jfd, SQLITE_SYNC_DATAONLY | pPager.syncFlags);

                // At this point the transaction is committed but the write lock is still held on the file. If there is a size limit configured for
                // the persistent journal and the journal file currently consumes more space than that limit allows for, truncate it now. There is no need
                // to sync the file following this operation.
                if (rc == SQLITE.OK && iLimit > 0)
                {
                    long sz = 0;
                    rc = sqlite3OsFileSize(pPager.jfd, ref sz);
                    if (rc == SQLITE.OK && sz > iLimit)
                        rc = sqlite3OsTruncate(pPager.jfd, iLimit);
                }
            }
            return rc;
        }

        // The journal file must be open when this routine is called. A journal header (JOURNAL_HDR_SZ bytes) is written into the journal file at the
        // current location.
        // The format for the journal header is as follows:
        // - 8 bytes: Magic identifying journal format.
        // - 4 bytes: Number of records in journal, or -1 no-sync mode is on.
        // - 4 bytes: Random number used for page hash.
        // - 4 bytes: Initial database page count.
        // - 4 bytes: Sector size used by the process that wrote this journal.
        // - 4 bytes: Database page size.
        // Followed by (JOURNAL_HDR_SZ - 28) bytes of unused space.
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
            if (pPager.noSync || (pPager.journalMode == JOURNALMODE.MEMORY) || (sqlite3OsDeviceCharacteristics(pPager.fd) & SQLITE_IOCAP_SAFE_APPEND) != 0)
            {
                aJournalMagic.CopyTo(zHeader, 0);
                put32bits(zHeader, aJournalMagic.Length, 0xffffffff);
            }
            else
                Array.Clear(zHeader, 0, aJournalMagic.Length + 4);
            // The random check-hash initialiser
            long i64Temp = 0;
            sqlite3_randomness(sizeof(long), ref i64Temp);
            pPager.cksumInit = (Pgno)i64Temp;
            put32bits(zHeader, aJournalMagic.Length + 4, pPager.cksumInit);
            // The initial database size
            put32bits(zHeader, aJournalMagic.Length + 8, pPager.dbOrigSize);
            // The assumed sector size for this process
            put32bits(zHeader, aJournalMagic.Length + 12, pPager.sectorSize);
            // The page size
            put32bits(zHeader, aJournalMagic.Length + 16, (uint)pPager.pageSize);
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
                IOTRACE("JHDR %p %lld %d\n", pPager, pPager.journalHdr, nHeader);
                rc = sqlite3OsWrite(pPager.jfd, zHeader, (int)nHeader, pPager.journalOff);
                Debug.Assert(pPager.journalHdr <= pPager.journalOff);
                pPager.journalOff += (int)nHeader;
            }
            return rc;
        }

        // The journal file must be open when this is called. A journal header file (JOURNAL_HDR_SZ bytes) is read from the current location in the journal
        // file. The current location in the journal file is given by pPager.journalOff. See comments above function writeJournalHdr() for
        // a description of the journal header format.
        //
        // If the header is read successfully, *pNRec is set to the number of page records following this header and *pDbSize is set to the size of the
        // database before the transaction began, in pages. Also, pPager.cksumInit is set to the value read from the journal header. SQLITE_OK is returned
        // in this case.
        //
        // If the journal header file appears to be corrupted, SQLITE_DONE is returned and *pNRec and *PDbSize are undefined.  If JOURNAL_HDR_SZ bytes
        // cannot be read from the journal file an error code is returned.
        internal static SQLITE readJournalHdr(Pager pPager, int isHot, long journalSize, out uint pNRec, out uint pDbSize)
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
                rc = sqlite3OsRead(pPager.jfd, aMagic, aMagic.Length, iHdrOff);
                if (rc != SQLITE.OK)
                    return rc;
                if (memcmp(aMagic, aJournalMagic, aMagic.Length) != 0)
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

        // Write the supplied master journal name into the journal file for pager pPager at the current location. The master journal name must be the last
        // thing written to a journal file. If the pager is in full-sync mode, the journal file descriptor is advanced to the next sector boundary before
        // anything is written. The format is:
        //   + 4 bytes: PAGER_MJ_PGNO.
        //   + N bytes: Master journal filename in utf-8.
        //   + 4 bytes: N (length of master journal name in bytes, no nul-terminator).
        //   + 4 bytes: Master journal name checksum.
        //   + 8 bytes: aJournalMagic[].
        // The master journal page checksum is the sum of the bytes in the master journal name, where each byte is interpreted as a signed 8-bit integer.
        // If zMaster is a NULL pointer (occurs for a single database transaction), this call is a no-op.
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
                || SQLITE.OK != (rc = sqlite3OsWrite(pPager.jfd, Encoding.UTF8.GetBytes(zMaster), nMaster, iHdrOff + 4))
                || SQLITE.OK != (rc = write32bits(pPager.jfd, iHdrOff + 4 + nMaster, (uint)nMaster))
                || SQLITE.OK != (rc = write32bits(pPager.jfd, iHdrOff + 4 + nMaster + 4, cksum))
                || SQLITE.OK != (rc = sqlite3OsWrite(pPager.jfd, aJournalMagic, 8, iHdrOff + 4 + nMaster + 8)))
                return rc;
            pPager.journalOff += nMaster + 20;
            // If the pager is in peristent-journal mode, then the physical journal-file may extend past the end of the master-journal name
            // and 8 bytes of magic data just written to the file. This is dangerous because the code to rollback a hot-journal file
            // will not be able to find the master-journal name to determine whether or not the journal is hot.
            // Easiest thing to do in this scenario is to truncate the journal file to the required size.
            long jrnlSize = 0;  // Size of journal file on disk
            if (SQLITE.OK == (rc = sqlite3OsFileSize(pPager.jfd, ref jrnlSize)) && jrnlSize > pPager.journalOff)
                rc = sqlite3OsTruncate(pPager.jfd, pPager.journalOff);
            return rc;
        }
    }
}
