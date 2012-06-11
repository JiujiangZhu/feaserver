using System;
using System.Diagnostics;
namespace Contoso.Sys
{
    public class MemJournalFile : VirtualFile
    {
        const int JOURNAL_CHUNKSIZE = 4096;

        public class FileChunk
        {
            public FileChunk pNext;                             // Next chunk in the journal 
            public byte[] zChunk = new byte[JOURNAL_CHUNKSIZE]; // Content of this chunk 
        }

        public class FilePoint
        {
            public long iOffset;           // Offset from the beginning of the file 
            public FileChunk pChunk;      // Specific chunk into which cursor points 
        }

        public FileChunk pFirst;              // Head of in-memory chunk-list
        public FilePoint endpoint = new FilePoint();            // Pointer to the end of the file
        public FilePoint readpoint = new FilePoint();           // Pointer to the end of the last xRead()

        internal MemJournalFile()
        {
            isOpen = true;
        }

        #region Methods

        public override int iVersion { get { return 1; } }

        public override RC xClose() { xTruncate(0); return RC.OK; }

        public override RC xRead(byte[] zBuf, int iAmt, long iOfst)
        {
            // SQLite never tries to read past the end of a rollback journal file 
            Debug.Assert(iOfst + iAmt <= endpoint.iOffset);
            FileChunk pChunk;
            if (readpoint.iOffset != iOfst || iOfst == 0)
            {
                var iOff = 0;
                for (pChunk = pFirst; Check.ALWAYS(pChunk != null) && (iOff + JOURNAL_CHUNKSIZE) <= iOfst; pChunk = pChunk.pNext)
                    iOff += JOURNAL_CHUNKSIZE;
            }
            else
                pChunk = readpoint.pChunk;
            var iChunkOffset = (int)(iOfst % JOURNAL_CHUNKSIZE);
            var izOut = 0;
            var nRead = iAmt;
            do
            {
                var iSpace = JOURNAL_CHUNKSIZE - iChunkOffset;
                var nCopy = Math.Min(nRead, JOURNAL_CHUNKSIZE - iChunkOffset);
                Buffer.BlockCopy(pChunk.zChunk, iChunkOffset, zBuf, izOut, nCopy);
                izOut += nCopy;
                nRead -= iSpace;
                iChunkOffset = 0;
            } while (nRead >= 0 && (pChunk = pChunk.pNext) != null && nRead > 0);
            readpoint.iOffset = (int)(iOfst + iAmt);
            readpoint.pChunk = pChunk;
            return RC.OK;
        }

        public override RC xWrite(byte[] zBuf, int iAmt, long iOfst)
        {
            SysEx.UNUSED_PARAMETER(iOfst);
            // An in-memory journal file should only ever be appended to. Random access writes are not required by sqlite.
            Debug.Assert(iOfst == endpoint.iOffset);
            var izWrite = 0;
            while (iAmt > 0)
            {
                var pChunk = endpoint.pChunk;
                var iChunkOffset = (int)(endpoint.iOffset % JOURNAL_CHUNKSIZE);
                var iSpace = Math.Min(iAmt, JOURNAL_CHUNKSIZE - iChunkOffset);
                if (iChunkOffset == 0)
                {
                    // New chunk is required to extend the file.
                    var pNew = new FileChunk();
                    if (pNew == null)
                        return RC.IOERR_NOMEM;
                    pNew.pNext = null;
                    if (pChunk != null) { Debug.Assert(pFirst != null); pChunk.pNext = pNew; }
                    else { Debug.Assert(pFirst == null); pFirst = pNew; }
                    endpoint.pChunk = pNew;
                }
                Buffer.BlockCopy(zBuf, izWrite, endpoint.pChunk.zChunk, iChunkOffset, iSpace);
                izWrite += iSpace;
                iAmt -= iSpace;
                endpoint.iOffset += iSpace;
            }
            return RC.OK;
        }

        public override RC xTruncate(long size)
        {
            SysEx.UNUSED_PARAMETER(size);
            Debug.Assert(size == 0);
            var pChunk = pFirst;
            while (pChunk != null)
            {
                var pTmp = pChunk;
                pChunk = pChunk.pNext;
            }
            // clear
            pFirst = null;
            endpoint = new FilePoint();
            readpoint = new FilePoint();
            return RC.OK;
        }

        public override RC xSync(SYNC NotUsed2) { SysEx.UNUSED_PARAMETER(NotUsed2); return RC.OK; }
        public override RC xFileSize(ref long pSize) { pSize = endpoint.iOffset; return RC.OK; }

        public override RC xLock(LOCK locktype) { return RC.OK; }
        public override RC xUnlock(LOCK locktype) { return RC.OK; }
        public override RC xCheckReservedLock(ref int pResOut) { return RC.OK; }
        public override RC xFileControl(FCNTL op, ref long pArg) { return RC.NOTFOUND; }
        public override int xSectorSize() { return (int)sectorSize; }
        public override IOCAP xDeviceCharacteristics() { return 0; }
        //
        public override RC xShmMap(int iPg, int pgsz, int pInt, out object pvolatile) { pvolatile = null; return RC.OK; }
        public override RC xShmLock(int offset, int n, int flags) { return RC.OK; }
        public override void xShmBarrier() { }
        public override RC xShmUnmap(int deleteFlag) { return RC.OK; }

        #endregion

        //internal static bool sqlite3IsMemJournal(VirtualFile pJfd) { return pJfd is MemJournalFile; }

        // Return the number of bytes required to store a MemJournal file descriptor.
        internal static int sqlite3MemJournalSize() { return 3096; }
    }
}
