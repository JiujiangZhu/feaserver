using System.IO;
using System.Text;
using System;
using System.Diagnostics;
using System.Threading;
using System.Runtime.InteropServices;
namespace Contoso.Sys
{
    public class VirtualFileSystem
    {
        const int SQLITE_DEFAULT_SECTOR_SIZE = 512;
        const int MAX_PATH = 260;
        static int MX_DELETION_ATTEMPTS = 5;

        public enum OPEN : uint
        {
            READONLY = 0x00000001,          // Ok for sqlite3_open_v2() 
            READWRITE = 0x00000002,         // Ok for sqlite3_open_v2() 
            CREATE = 0x00000004,            // Ok for sqlite3_open_v2() 
            DELETEONCLOSE = 0x00000008,     // VFS only 
            EXCLUSIVE = 0x00000010,         // VFS only 
            AUTOPROXY = 0x00000020,         // VFS only 
            URI = 0x00000040,               // Ok for sqlite3_open_v2() 
            MAIN_DB = 0x00000100,           // VFS only 
            TEMP_DB = 0x00000200,           // VFS only 
            TRANSIENT_DB = 0x00000400,      // VFS only 
            MAIN_JOURNAL = 0x00000800,      // VFS only 
            TEMP_JOURNAL = 0x00001000,      // VFS only 
            SUBJOURNAL = 0x00002000,        // VFS only 
            MASTER_JOURNAL = 0x00004000,    // VFS only 
            NOMUTEX = 0x00008000,           // Ok for sqlite3_open_v2() 
            FULLMUTEX = 0x00010000,         // Ok for sqlite3_open_v2() 
            SHAREDCACHE = 0x00020000,       // Ok for sqlite3_open_v2() 
            PRIVATECACHE = 0x00040000,      // Ok for sqlite3_open_v2() 
            WAL = 0x00080000,               // VFS only 
        }

        public enum ACCESS
        {
            EXISTS = 0,
            READWRITE = 1, // Used by PRAGMA temp_store_directory
            READ = 2, // Unused
        }

        public int iVersion = -3;            // Structure version number (currently 3)
        public int szOsFile = -1;            // Size of subclassed VirtualFile
        public int mxPathname = MAX_PATH;          // Maximum file pathname length
        public VirtualFileSystem pNext;       // Next registered VFS
        public string zName = "win32";            // Name of this virtual file system
        public object pAppData = 0;         // Pointer to application-specific data

        public VirtualFileSystem() { }
        public VirtualFileSystem(int iVersion, int szOsFile, int mxPathname, VirtualFileSystem pNext, string zName, object pAppData)
        {
            this.iVersion = iVersion;
            this.szOsFile = szOsFile;
            this.mxPathname = mxPathname;
            this.pNext = pNext;
            this.zName = zName;
            this.pAppData = pAppData;
        }

        public void CopyTo(VirtualFileSystem ct)
        {
            ct.iVersion = this.iVersion;
            ct.szOsFile = this.szOsFile;
            ct.mxPathname = this.mxPathname;
            ct.pNext = this.pNext;
            ct.zName = this.zName;
            ct.pAppData = this.pAppData;
        }

        public SQLITE xOpen(string zName, VirtualFile pFile, OPEN flags, out OPEN pOutFlags)
        {
            pOutFlags = 0;
            // If argument zPath is a NULL pointer, this function is required to open a temporary file. Use this buffer to store the file name in.
            //var zTmpname = new StringBuilder(MAX_PATH + 1);        // Buffer used to create temp filename
            var rc = SQLITE.OK;
            var eType = (OPEN)((int)flags & 0xFFFFFF00);  // Type of file to open
            var isExclusive = (flags & OPEN.EXCLUSIVE) != 0;
            var isDelete = (flags & OPEN.DELETEONCLOSE) != 0;
            var isCreate = (flags & OPEN.CREATE) != 0;
            var isReadonly = (flags & OPEN.READONLY) != 0;
            var isReadWrite = (flags & OPEN.READWRITE) != 0;
            var isOpenJournal = (isCreate && (eType == OPEN.MASTER_JOURNAL || eType == OPEN.MAIN_JOURNAL || eType == OPEN.WAL));
            // Check the following statements are true:
            //   (a) Exactly one of the READWRITE and READONLY flags must be set, and
            //   (b) if CREATE is set, then READWRITE must also be set, and
            //   (c) if EXCLUSIVE is set, then CREATE must also be set.
            //   (d) if DELETEONCLOSE is set, then CREATE must also be set.
            Debug.Assert((!isReadonly || !isReadWrite) && (isReadWrite || isReadonly));
            Debug.Assert(!isCreate || isReadWrite);
            Debug.Assert(!isExclusive || isCreate);
            Debug.Assert(!isDelete || isCreate);
            // The main DB, main journal, WAL file and master journal are never automatically deleted. Nor are they ever temporary files.
            //Debug.Assert((!isDelete && !string.IsNullOrEmpty(zName)) || eType != OPEN.MAIN_DB);
            Debug.Assert((!isDelete && !string.IsNullOrEmpty(zName)) || eType != OPEN.MAIN_JOURNAL);
            Debug.Assert((!isDelete && !string.IsNullOrEmpty(zName)) || eType != OPEN.MASTER_JOURNAL);
            Debug.Assert((!isDelete && !string.IsNullOrEmpty(zName)) || eType != OPEN.WAL);
            // Assert that the upper layer has set one of the "file-type" flags.
            Debug.Assert(eType == OPEN.MAIN_DB || eType == OPEN.TEMP_DB || eType == OPEN.MAIN_JOURNAL || eType == OPEN.TEMP_JOURNAL ||
                eType == OPEN.SUBJOURNAL || eType == OPEN.MASTER_JOURNAL || eType == OPEN.TRANSIENT_DB || eType == OPEN.WAL);
            pFile.fs = null;
            // If the second argument to this function is NULL, generate a temporary file name to use
            if (string.IsNullOrEmpty(zName))
            {
                Debug.Assert(isDelete && !isOpenJournal);
                zName = Path.GetRandomFileName();
            }
            // Convert the filename to the system encoding.
            if (zName.StartsWith("/") && !zName.StartsWith("//"))
                zName = zName.Substring(1);
            var dwDesiredAccess = (isReadWrite ? FileAccess.Read | FileAccess.Write : FileAccess.Read);
            // SQLITE_OPEN_EXCLUSIVE is used to make sure that a new file is created. SQLite doesn't use it to indicate "exclusive access" as it is usually understood.
            FileMode dwCreationDisposition;
            if (isExclusive)
                // Creates a new file, only if it does not already exist. */ If the file exists, it fails.
                dwCreationDisposition = FileMode.CreateNew;
            else if (isCreate)
                // Open existing file, or create if it doesn't exist
                dwCreationDisposition = FileMode.OpenOrCreate;
            else
                // Opens a file, only if it exists.
                dwCreationDisposition = FileMode.Open;
            var dwShareMode = FileShare.Read | FileShare.Write;
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
            FileOptions dwFlagsAndAttributes;
#endif
            if (isDelete)
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                dwFlagsAndAttributes = FileOptions.DeleteOnClose;
#endif
            else
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                dwFlagsAndAttributes = FileOptions.None;
#endif
            // Reports from the internet are that performance is always better if FILE_FLAG_RANDOM_ACCESS is used.
            FileStream fs = null;
            if (Environment.OSVersion.Platform >= PlatformID.Win32NT)
            {
                // retry opening the file a few times; this is because of a racing condition between a delete and open call to the FS
                var retries = 3;
                while (fs == null && retries > 0)
                    try
                    {
                        retries--;
#if WINDOWS_PHONE || SQLITE_SILVERLIGHT  
                        fs = new IsolatedStorageFileStream(zConverted, dwCreationDisposition, dwDesiredAccess, dwShareMode, IsolatedStorageFile.GetUserStoreForApplication());
#elif !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                        fs = new FileStream(zName, dwCreationDisposition, dwDesiredAccess, dwShareMode, 4096, dwFlagsAndAttributes);
#else
                        fs = new FileStream(zName, dwCreationDisposition, dwDesiredAccess, dwShareMode, 4096);
#endif
#if DEBUG
                        SysEx.OSTRACE("OPEN %d (%s)\n", fs.GetHashCode(), fs.Name);
#endif
                    }
                    catch (Exception) { Thread.Sleep(100); }
            }
            SysEx.OSTRACE("OPEN %d %s 0x%lx %s\n", pFile.GetHashCode(), zName, dwDesiredAccess, fs == null ? "failed" : "ok");
            if (fs == null ||
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
 fs.SafeFileHandle.IsInvalid
#else
 !fs.CanRead
#endif
)
            {
#if SQLITE_SILVERLIGHT
pFile.lastErrno = 1;
#else
                pFile.lastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                VirtualFile.winLogError(SQLITE.CANTOPEN, "winOpen", zName);
                if (isReadWrite)
                    return xOpen(zName, pFile, ((flags | OPEN.READONLY) & ~(OPEN.CREATE | OPEN.READWRITE)), out pOutFlags);
                else
                    return SysEx.SQLITE_CANTOPEN_BKPT();
            }
            pOutFlags = (isReadWrite ? OPEN.READWRITE : OPEN.READONLY);
            pFile.Clear();
            pFile.isOpen = true;
            pFile.fs = fs;
            pFile.lastErrno = 0;
            pFile.pVfs = this;
            pFile.pShm = null;
            pFile.zPath = zName;
            pFile.sectorSize = (ulong)getSectorSize(zName);
            return rc;
        }

        private ulong getSectorSize(string zName) { return SQLITE_DEFAULT_SECTOR_SIZE; }

        public SQLITE xDelete(string zName, int syncDir)
        {
            int cnt = 0;
            int error;
            SysEx.UNUSED_PARAMETER(syncDir);
            SQLITE rc;
            if (Environment.OSVersion.Platform >= PlatformID.Win32NT)
            {
                do
                {
#if WINDOWS_PHONE
           if ( !System.IO.IsolatedStorage.IsolatedStorageFile.GetUserStoreForApplication().FileExists( zFilename ) )
#elif SQLITE_SILVERLIGHT
            if (!IsolatedStorageFile.GetUserStoreForApplication().FileExists(zFilename))
#else
                    if (!File.Exists(zName))
#endif
                    { rc = SQLITE.IOERR; break; }
                    try
                    {
#if WINDOWS_PHONE
              System.IO.IsolatedStorage.IsolatedStorageFile.GetUserStoreForApplication().DeleteFile(zFilename);
#elif SQLITE_SILVERLIGHT
              IsolatedStorageFile.GetUserStoreForApplication().DeleteFile(zFilename);
#else
                        File.Delete(zName);
#endif
                        rc = SQLITE.OK;
                    }
                    catch (IOException e) { rc = SQLITE.IOERR; Thread.Sleep(100); }
                } while (rc != SQLITE.OK && ++cnt < MX_DELETION_ATTEMPTS);
                // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed. Since the ASCII version of these Windows API do not exist for WINCE,
                // it's important to not reference them for WINCE builds.
#if !SQLITE_OS_WINCE
            }
            else
            {
                do
                {
                    if (!File.Exists(zName)) { rc = SQLITE.IOERR; break; }
                    try { File.Delete(zName); rc = SQLITE.OK; }
                    catch (IOException e) { rc = SQLITE.IOERR; Thread.Sleep(100); }
                } while (rc != SQLITE.OK && cnt++ < MX_DELETION_ATTEMPTS);
#endif
            }
#if DEBUG
            SysEx.OSTRACE("DELETE \"%s\"\n", zName);
#endif
            if (rc == SQLITE.OK)
                return rc;
#if SQLITE_SILVERLIGHT
            error = (int)ERROR_NOT_SUPPORTED;
#else
            error = Marshal.GetLastWin32Error();
#endif
            return (rc == SQLITE.INVALID && error == VirtualFile.ERROR_FILE_NOT_FOUND ? SQLITE.OK : VirtualFile.winLogError(SQLITE.IOERR_DELETE, "winDelete", zName));
        }

        public SQLITE xAccess(string zName, ACCESS flags, out int pResOut)
        {
            var rc = SQLITE.OK;
            // Do a quick test to prevent the try/catch block
            if (flags == ACCESS.EXISTS)
            {
#if WINDOWS_PHONE
          pResOut = (System.IO.IsolatedStorage.IsolatedStorageFile.GetUserStoreForApplication().FileExists(zFilename) ? 1 : 0);
#elif SQLITE_SILVERLIGHT
          pResOut = (IsolatedStorageFile.GetUserStoreForApplication().FileExists(zFilename) ? 1 : 0);
#else
                pResOut = (File.Exists(zName) ? 1 : 0);
#endif
                return SQLITE.OK;
            }
            FileAttributes attr = 0;
            try
            {
#if WINDOWS_PHONE || WINDOWS_MOBILE || SQLITE_SILVERLIGHT
        if (new DirectoryInfo(zFilename).Exists)
#else
                attr = File.GetAttributes(zName);
                if (attr == FileAttributes.Directory)
#endif
                    try
                    {
                        var name = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());
                        var fs = File.Create(name);
                        fs.Close();
                        File.Delete(name);
                        attr = FileAttributes.Normal;
                    }
                    catch (IOException) { attr = FileAttributes.ReadOnly; }
            }
            // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed. Since the ASCII version of these Windows API do not exist for WINCE,
            // it's important to not reference them for WINCE builds.
            catch (IOException) { VirtualFile.winLogError(SQLITE.IOERR_ACCESS, "winAccess", zName); }
            switch (flags)
            {
                case ACCESS.READ:
                case ACCESS.EXISTS: rc = (attr != 0 ? SQLITE.ERROR : SQLITE.OK); break;
                case ACCESS.READWRITE: rc = (attr == 0 ? SQLITE.OK : (attr & FileAttributes.ReadOnly) != 0 ? SQLITE.OK : SQLITE.ERROR); break;
                default: Debug.Assert("" == "Invalid flags argument"); rc = SQLITE.OK; break;
            }
            pResOut = (int)rc;
            return SQLITE.OK;
        }

        public SQLITE xFullPathname(string zName, int nOut, StringBuilder zOut)
        {
            if (zName[0] == '/' && Char.IsLetter(zName[1]) && zName[2] == ':')
                zName = zName.Substring(1);
            string path;
            try { path = Path.GetFullPath(zName); }
            catch (Exception) { path = zName; }
            if (path != null)
            {
                if (zOut.Length > mxPathname)
                    zOut.Length = mxPathname;
                zOut.Append(path);
                return SQLITE.OK;
            }
            return SQLITE.NOMEM;
        }

        // TODO -- Fix This
        public IntPtr xDlOpen(string zFilename) { return IntPtr.Zero; }
        public SQLITE xDlError(int nByte, string zErrMsg) { return SQLITE.OK; }
        public IntPtr xDlSym(IntPtr data, string zSymbol) { return IntPtr.Zero; }
        public SQLITE xDlClose(IntPtr data) { return SQLITE.OK; }

        public int xRandomness(int nByte, byte[] buffer)
        {
            var sBuf = BitConverter.GetBytes(DateTime.Now.Ticks);
            buffer[0] = sBuf[0];
            buffer[1] = sBuf[1];
            buffer[2] = sBuf[2];
            buffer[3] = sBuf[3];
            var n = 16;
            if (sizeof(ulong) <= nByte - n)
            {
                uint processId;
#if !SQLITE_SILVERLIGHT
                processId = (uint)Process.GetCurrentProcess().Id;
#else
processId = 28376023;
#endif
                ConvertEx.put32bits(buffer, n, processId);
                n += 4;
            }
            if (sizeof(ulong) <= nByte - n)
            {
                uint i = (uint)new DateTime().Ticks;
                ConvertEx.put32bits(buffer, n, i);
                n += 4;
            }
            if (sizeof(long) <= nByte - n)
            {
                long i = DateTime.UtcNow.Millisecond;
                ConvertEx.put32bits(buffer, n, (uint)(i & 0xFFFFFFFF));
                ConvertEx.put32bits(buffer, n, (uint)(i >> 32));
                n += sizeof(long);
            }
            return n;
        }

        public int xSleep(int microseconds)
        {
            var millisecondsTimeout = ((microseconds + 999) / 1000);
            Thread.Sleep(millisecondsTimeout);
            return millisecondsTimeout * 1000;
        }

        public SQLITE xCurrentTime(ref double currenttime)
        {
            long i = 0;
            var rc = xCurrentTimeInt64(ref i);
            if (rc == SQLITE.OK)
                currenttime = i / 86400000.0;
            return rc;
        }

        public SQLITE xGetLastError(int nBuf, ref string zBuf)
        {
#if SQLITE_SILVERLIGHT
            zBuf = "Unknown error";
#else
            zBuf = Marshal.GetLastWin32Error().ToString();
#endif
            return SQLITE.OK;
        }

        public SQLITE xCurrentTimeInt64(ref long pTime)
        {
            const long winFiletimeEpoch = 23058135 * (long)8640000;
            pTime = winFiletimeEpoch + DateTime.UtcNow.ToFileTimeUtc() / (long)10000;
            return SQLITE.OK;
        }

        public SQLITE xSetSystemCall(string zName, long sqlite3_syscall_ptr) { return SQLITE.OK; }
        public SQLITE xGetSystemCall(string zName, long sqlite3_syscall_ptr) { return SQLITE.OK; }
        public SQLITE xNextSystemCall(string zName, long sqlite3_syscall_ptr) { return SQLITE.OK; }























    }
}
