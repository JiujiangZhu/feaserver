using System;
using System.IO;
namespace Contoso.Sys
{
    public class sqlite3_file : sqlite3_io_methods
    {
        public bool isOpen;
        public sqlite3_vfs pVfs;        // The VFS used to open this file
        public FileStream fs;           // Filestream access to this file
        // public HANDLE h;             // Handle for accessing the file
        public int locktype;            // Type of lock currently held on this file
        public int sharedLockByte;      // Randomly chosen byte used as a shared lock
        public ulong lastErrno;         // The Windows errno from the last I/O error
        public ulong sectorSize;        // Sector size of the device file is on
        //#if !SQLITE_OMIT_WAL           
        //public winShm pShm;           // Instance of shared memory on this file
        //#else
        public object pShm;             // DUMMY Instance of shared memory on this file
        //#endif
        public string zPath;            // Full pathname of this file
        public int szChunk;             // Chunk size configured by FCNTL_CHUNK_SIZE

        public void Clear()
        {
            fs = null;
            locktype = 0;
            sharedLockByte = 0;
            lastErrno = 0;
            sectorSize = 0;
        }

        public int iVersion
        {
            get { throw new NotImplementedException(); }
        }

        public int xClose(sqlite3_file File_ID)
        {
            throw new NotImplementedException();
        }

        public int xRead(sqlite3_file File_ID, byte[] buffer, int amount, long offset)
        {
            throw new NotImplementedException();
        }

        public int xWrite(sqlite3_file File_ID, byte[] buffer, int amount, long offset)
        {
            throw new NotImplementedException();
        }

        public int xTruncate(sqlite3_file File_ID, long size)
        {
            throw new NotImplementedException();
        }

        public int xSync(sqlite3_file File_ID, int flags)
        {
            throw new NotImplementedException();
        }

        public int xFileSize(sqlite3_file File_ID, ref long size)
        {
            throw new NotImplementedException();
        }

        public int xLock(sqlite3_file File_ID, int locktype)
        {
            throw new NotImplementedException();
        }

        public int xUnlock(sqlite3_file File_ID, int locktype)
        {
            throw new NotImplementedException();
        }

        public int xCheckReservedLock(sqlite3_file File_ID, ref int pRes)
        {
            throw new NotImplementedException();
        }

        public int xFileControl(sqlite3_file File_ID, int op, ref long pArgs)
        {
            throw new NotImplementedException();
        }

        public int xSectorSize(sqlite3_file File_ID)
        {
            throw new NotImplementedException();
        }

        public int xDeviceCharacteristics(sqlite3_file File_ID)
        {
            throw new NotImplementedException();
        }

        public int xShmMap(sqlite3_file File_ID, int iPg, int pgsz, int pInt, out object pvolatile)
        {
            throw new NotImplementedException();
        }

        public int xShmLock(sqlite3_file File_ID, int offset, int n, int flags)
        {
            throw new NotImplementedException();
        }

        public void xShmBarrier(sqlite3_file File_ID)
        {
            throw new NotImplementedException();
        }

        public int xShmUnmap(sqlite3_file File_ID, int deleteFlag)
        {
            throw new NotImplementedException();
        }
    }
}
