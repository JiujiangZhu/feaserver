using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
namespace Contoso.Sys
{
    public partial class VirtualFile
    {
        public static int MX_CLOSE_ATTEMPT = 3;
        internal const long ERROR_FILE_NOT_FOUND = 2L;
        internal const long ERROR_HANDLE_DISK_FULL = 39L;
        internal const long ERROR_NOT_SUPPORTED = 50L;
        internal const long ERROR_DISK_FULL = 112L;
        internal static int PENDING_BYTE = 0x40000000;
        private static int RESERVED_BYTE = (PENDING_BYTE + 1);
        private static int SHARED_FIRST = (PENDING_BYTE + 2);
        private static int SHARED_SIZE = 510;

        public enum LOCK : byte
        {
            NO = 0,
            SHARED = 1,
            RESERVED = 2,
            PENDING = 3,
            EXCLUSIVE = 4,
            UNKNOWN = 5,
        }

        [Flags]
        public enum SYNC : byte
        {
            NORMAL = 0x00002,
            FULL = 0x00003,
            DATAONLY = 0x00010,
        }

        public enum FCNTL : uint
        {
            LOCKSTATE = 1,
            GET_LOCKPROXYFILE = 2,
            SET_LOCKPROXYFILE = 3,
            LAST_ERRNO = 4,
            SIZE_HINT = 5,
            CHUNK_SIZE = 6,
            FILE_POINTER = 7,
            SYNC_OMITTED = 8,
            DB_UNCHANGED = 0xca093fa0,
        }

        [Flags]
        public enum IOCAP : uint
        {
            ATOMIC = 0x00000001,
            ATOMIC512 = 0x00000002,
            ATOMIC1K = 0x00000004,
            ATOMIC2K = 0x00000008,
            ATOMIC4K = 0x00000010,
            ATOMIC8K = 0x00000020,
            ATOMIC16K = 0x00000040,
            ATOMIC32K = 0x00000080,
            ATOMIC64K = 0x00000100,
            SAFE_APPEND = 0x00000200,
            SEQUENTIAL = 0x00000400,
            UNDELETABLE_WHEN_OPEN = 0x00000800,
        }

        public bool isOpen;
        public VirtualFileSystem pVfs;        // The VFS used to open this file
        public FileStream fs;           // Filestream access to this file
        // public HANDLE h;             // Handle for accessing the file
        public LOCK locktype;            // Type of lock currently held on this file
        public int sharedLockByte;      // Randomly chosen byte used as a shared lock
        public ulong lastErrno;         // The Windows errno from the last I/O error
        public ulong sectorSize;        // Sector size of the device file is on
#if !SQLITE_OMIT_WAL           
        public winShm pShm;           // Instance of shared memory on this file
#else
        public object pShm;             // DUMMY Instance of shared memory on this file
#endif
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

        #region Methods

        public virtual int iVersion
        {
            get { return 2; }
        }

        public virtual RC xClose()
        {
            Debug.Assert(pShm == null);
#if SQLITE_DEBUG
            SysEx.OSTRACE("CLOSE {0} ({1})", fs.GetHashCode(), fs.Name);
#endif
            bool rc;
            int cnt = 0;
            do
            {
                fs.Close();
                rc = true;
            } while (!rc && ++cnt < MX_CLOSE_ATTEMPT);
            return rc ? RC.OK : winLogError(RC.IOERR_CLOSE, "winClose", zPath);
        }

        public virtual RC xRead(byte[] buffer, int amount, long offset)
        {
            if (buffer == null) buffer = new byte[amount]; 
#if DEBUG
            SysEx.OSTRACE("READ {0} lock={1}", fs.GetHashCode(), locktype);
#endif
            if (!fs.CanRead)
                return RC.IOERR_READ;
            if (seekWinFile(offset) != 0)
                return RC.FULL;
            int nRead; // Number of bytes actually read from file
            try { nRead = fs.Read(buffer, 0, amount); }
            catch (Exception) { lastErrno = (uint)Marshal.GetLastWin32Error(); return winLogError(RC.IOERR_READ, "winRead", zPath); }
            if (nRead < amount)
            {
                // Unread parts of the buffer must be zero-filled
                Array.Clear(buffer, (int)nRead, (int)(amount - nRead));
                return RC.IOERR_SHORT_READ;
            }
            return RC.OK;
        }

        public virtual RC xWrite(byte[] buffer, int amount, long offset)
        {
            Debug.Assert(amount > 0);
#if DEBUG
            SysEx.OSTRACE("WRITE {0} lock={1}", fs.GetHashCode(), locktype);
#endif
            var rc = seekWinFile(offset);
            long wrote = fs.Position;
            try
            {
                Debug.Assert(buffer.Length >= amount);
                fs.Write(buffer, 0, amount);
                rc = 1;
                wrote = fs.Position - wrote;
            }
            catch (IOException) { return RC.READONLY; }
            if (rc == 0 || amount > (int)wrote)
            {
                lastErrno = (uint)Marshal.GetLastWin32Error();
                return (lastErrno == ERROR_HANDLE_DISK_FULL || lastErrno == ERROR_DISK_FULL ? RC.FULL : winLogError(RC.IOERR_WRITE, "winWrite", zPath));
            }
            return RC.OK;
        }

        public virtual RC xTruncate(long size)
        {
            var rc = RC.OK;
#if DEBUG
            SysEx.OSTRACE("TRUNCATE {0} {1,ll}", fs.Name, size);
#endif
            // If the user has configured a chunk-size for this file, truncate the file so that it consists of an integer number of chunks (i.e. the
            // actual file size after the operation may be larger than the requested size).
            if (szChunk != 0)
                size = ((size + szChunk - 1) / szChunk) * szChunk;
            try { fs.SetLength(size); rc = RC.OK; }
            catch (IOException e) { lastErrno = (uint)Marshal.GetLastWin32Error(); rc = winLogError(RC.IOERR_TRUNCATE, "winTruncate2", zPath); }
            SysEx.OSTRACE("TRUNCATE {0} {1,%ll} {2}", fs.GetHashCode(), size, rc == RC.OK ? "ok" : "failed");
            return rc;
        }

        public virtual RC xSync(SYNC flags)
        {
            // Check that one of SQLITE_SYNC_NORMAL or FULL was passed 
            Debug.Assert(((int)flags & 0x0F) == (int)SYNC.NORMAL || ((int)flags & 0x0F) == (int)SYNC.FULL);
            SysEx.OSTRACE("SYNC {0} lock={1}", fs.GetHashCode(), locktype);
#if SQLITE_NO_SYNC
return SQLITE.OK;
#else
            fs.Flush();
            return RC.OK;
#endif
        }

        public virtual RC xFileSize(ref long size)
        {
            size = (fs.CanRead ? fs.Length : 0);
            return RC.OK;
        }

        public virtual RC xLock(LOCK locktype)
        {
#if DEBUG
            SysEx.OSTRACE("LOCK {0} {1} was {2}({3})", fs.GetHashCode(), locktype, this.locktype, sharedLockByte);
#endif
            // If there is already a lock of this type or more restrictive on the OsFile, do nothing. Don't use the end_lock: exit path, as sqlite3OsEnterMutex() hasn't been called yet.
            if (this.locktype >= locktype)
                return RC.OK;
            // Make sure the locking sequence is correct
            Debug.Assert(this.locktype != LOCK.NO || locktype == LOCK.SHARED);
            Debug.Assert(locktype != LOCK.PENDING);
            Debug.Assert(locktype != LOCK.RESERVED || this.locktype == LOCK.SHARED);
            // Lock the PENDING_LOCK byte if we need to acquire a PENDING lock or a SHARED lock.  If we are acquiring a SHARED lock, the acquisition of
            // the PENDING_LOCK byte is temporary.
            uint error = 0;
            var res = 1;                // Result of a windows lock call
            var gotPendingLock = false; // True if we acquired a PENDING lock this time
            var newLocktype = this.locktype;
            if (this.locktype == LOCK.NO || (locktype == LOCK.EXCLUSIVE && this.locktype == LOCK.RESERVED))
            {
                res = 0;
                var cnt = 3;
                while (cnt-- > 0 && res == 0)
                    try { lockingStrategy.LockFile(this, PENDING_BYTE, 1); res = 1; }
                    catch (Exception)
                    {
                        // Try 3 times to get the pending lock.  The pending lock might be held by another reader process who will release it momentarily.
#if DEBUG
                        SysEx.OSTRACE("could not get a PENDING lock. cnt={0}", cnt);
#endif
                        Thread.Sleep(1);
                    }
                gotPendingLock = (res != 0);
                if (0 == res)
                    error = (uint)Marshal.GetLastWin32Error();
            }
            // Acquire a shared lock
            if (locktype == LOCK.SHARED && res != 0)
            {
                Debug.Assert(this.locktype == LOCK.NO);
                res = getReadLock();
                if (res != 0)
                    newLocktype = LOCK.SHARED;
                else
                    error = (uint)Marshal.GetLastWin32Error();
            }
            // Acquire a RESERVED lock
            if (locktype == LOCK.RESERVED && res != 0)
            {
                Debug.Assert(this.locktype == LOCK.SHARED);
                try { lockingStrategy.LockFile(this, RESERVED_BYTE, 1); newLocktype = LOCK.RESERVED; res = 1; }
                catch (Exception) { res = 0; error = (uint)Marshal.GetLastWin32Error(); }
                if (res != 0)
                    newLocktype = LOCK.RESERVED;
                else
                    error = (uint)Marshal.GetLastWin32Error();
            }
            // Acquire a PENDING lock
            if (locktype == LOCK.EXCLUSIVE && res != 0)
            {
                newLocktype = LOCK.PENDING;
                gotPendingLock = false;
            }
            // Acquire an EXCLUSIVE lock
            if (locktype == LOCK.EXCLUSIVE && res != 0)
            {
                Debug.Assert(this.locktype >= LOCK.SHARED);
                res = unlockReadLock();
#if DEBUG
                SysEx.OSTRACE("unreadlock = {0}", res);
#endif
                try { lockingStrategy.LockFile(this, SHARED_FIRST, SHARED_SIZE); newLocktype = LOCK.EXCLUSIVE; res = 1; }
                catch (Exception) { res = 0; }
                if (res != 0)
                    newLocktype = LOCK.EXCLUSIVE;
                else
                {
                    error = (uint)Marshal.GetLastWin32Error();
#if DEBUG
                    SysEx.OSTRACE("error-code = {0}", error);
#endif
                    getReadLock();
                }
            }
            // If we are holding a PENDING lock that ought to be released, then release it now.
            if (gotPendingLock && locktype == LOCK.SHARED)
                lockingStrategy.UnlockFile(this, PENDING_BYTE, 1);
            // Update the state of the lock has held in the file descriptor then return the appropriate result code.
            var rc = RC.OK;
            if (res != 0)
                rc = RC.OK;
            else
            {
#if DEBUG
                SysEx.OSTRACE("LOCK FAILED {0} trying for {1} but got {2}", fs.GetHashCode(), locktype, newLocktype);
#endif
                lastErrno = error;
                rc = RC.BUSY;
            }
            this.locktype = newLocktype;
            return rc;
        }

        public virtual RC xUnlock(LOCK locktype)
        {
            Debug.Assert(locktype <= LOCK.SHARED);
#if DEBUG
            SysEx.OSTRACE("UNLOCK {0} to {1} was {2}({3})", fs.GetHashCode(), locktype, this.locktype, sharedLockByte);
#endif
            var rc = RC.OK;
            var type = this.locktype;
            if (type >= LOCK.EXCLUSIVE)
            {
                lockingStrategy.UnlockFile(this, SHARED_FIRST, SHARED_SIZE);
                if (locktype == LOCK.SHARED && getReadLock() == 0)
                    // This should never happen.  We should always be able to reacquire the read lock 
                    rc = winLogError(RC.IOERR_UNLOCK, "winUnlock", zPath);
            }
            if (type >= LOCK.RESERVED)
                try { lockingStrategy.UnlockFile(this, RESERVED_BYTE, 1); }
                catch (Exception) { }
            if (locktype == LOCK.NO && type >= LOCK.SHARED)
                unlockReadLock();
            if (type >= LOCK.PENDING)
            {
                try { lockingStrategy.UnlockFile(this, PENDING_BYTE, 1); }
                catch (Exception) { }
            }
            this.locktype = locktype;
            return rc;
        }

        public virtual RC xCheckReservedLock(ref int pResOut)
        {
            int rc;
            if (locktype >= LOCK.RESERVED)
            {
                rc = 1;
#if DEBUG
                SysEx.OSTRACE("TEST WR-LOCK {0} {1} (local)", fs.Name, rc);
#endif
            }
            else
            {
                try
                {
                    lockingStrategy.LockFile(this, RESERVED_BYTE, 1);
                    lockingStrategy.UnlockFile(this, RESERVED_BYTE, 1);
                    rc = 1;
                }
                catch (IOException) { rc = 0; }
                rc = 1 - rc;
#if DEBUG
                SysEx.OSTRACE("TEST WR-LOCK {0} {1} (remote)", fs.GetHashCode(), rc);
#endif
            }
            pResOut = rc;
            return RC.OK;
        }

        public virtual RC xFileControl(FCNTL op, ref long pArg)
        {
            switch (op)
            {
                case FCNTL.LOCKSTATE:
                    pArg = (int)locktype;
                    return RC.OK;
                case FCNTL.LAST_ERRNO:
                    pArg = (int)lastErrno;
                    return RC.OK;
                case FCNTL.CHUNK_SIZE:
                    szChunk = (int)pArg;
                    return RC.OK;
                case FCNTL.SIZE_HINT:
                    var sz = (long)pArg;
                    xTruncate(sz);
                    return RC.OK;
                case FCNTL.SYNC_OMITTED:
                    return RC.OK;
            }
            return RC.NOTFOUND;
        }

        public virtual int xSectorSize() { return (int)sectorSize; }

        public virtual IOCAP xDeviceCharacteristics() { return 0; }

        #endregion

        internal static RC winLogError(RC a, string b, string c) { var sf = new StackTrace(new StackFrame(true)).GetFrame(0); return winLogErrorAtLine(a, b, c, sf.GetFileLineNumber()); }
        private static RC winLogErrorAtLine(RC errcode, string zFunc, string zPath, int iLine)
        {
            var iErrno = (uint)Marshal.GetLastWin32Error();
            var zMsg = Marshal.GetLastWin32Error().ToString();
            Debug.Assert(errcode != RC.OK);
            if (zPath == null)
                zPath = string.Empty;
            int i;
            for (i = 0; i < zMsg.Length && zMsg[i] != '\r' && zMsg[i] != '\n'; i++) ;
            zMsg = zMsg.Substring(0, i);
            //sqlite3_log(errcode, "os_win.c:%d: (%d) %s(%s) - %s", iLine, iErrno, zFunc, zPath, zMsg);
            return errcode;
        }

        private int seekWinFile(long iOffset)
        {
            try { fs.Seek(iOffset, SeekOrigin.Begin); }
            catch (Exception e) { lastErrno = (uint)Marshal.GetLastWin32Error(); winLogError(RC.IOERR_SEEK, "seekWinFile", zPath); return 1; }
            return 0;
        }

        private int getReadLock()
        {
            var res = 0;
            if (Environment.OSVersion.Platform >= PlatformID.Win32NT)
                res = lockingStrategy.SharedLockFile(this, SHARED_FIRST, SHARED_SIZE);
            // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed.
            if (res == 0)
                lastErrno = (uint)Marshal.GetLastWin32Error();
            // No need to log a failure to lock 
            return res;
        }

        private int unlockReadLock()
        {
            var res = 1;
            if (Environment.OSVersion.Platform >= PlatformID.Win32NT)
                try { lockingStrategy.UnlockFile(this, SHARED_FIRST, SHARED_SIZE); }
                catch (Exception) { res = 0; }
            if (res == 0) { lastErrno = (uint)Marshal.GetLastWin32Error(); winLogError(RC.IOERR_UNLOCK, "unlockReadLock", zPath); }
            return res;
        }
    }
}
