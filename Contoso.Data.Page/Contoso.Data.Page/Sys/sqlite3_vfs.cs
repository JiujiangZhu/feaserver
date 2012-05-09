using System.IO;
using System.Text;
using System;
namespace Contoso.Sys
{
    public abstract class sqlite3_vfs
    {
        public int iVersion;            // Structure version number (currently 3)
        public int szOsFile;            // Size of subclassed sqlite3_file
        public int mxPathname;          // Maximum file pathname length
        public sqlite3_vfs pNext;       // Next registered VFS
        public string zName;            // Name of this virtual file system
        public object pAppData;         // Pointer to application-specific data
        public abstract int xOpen(sqlite3_vfs vfs, string zName, sqlite3_file db, int flags, out int pOutFlags);
        public abstract int xDelete(sqlite3_vfs vfs, string zName, int syncDir);
        public abstract int xAccess(sqlite3_vfs vfs, string zName, int flags, out int pResOut);
        public abstract int xFullPathname(sqlite3_vfs vfs, string zName, int nOut, StringBuilder zOut);
        public abstract IntPtr xDlOpen(sqlite3_vfs vfs, string zFilename);
        public abstract int xDlError(sqlite3_vfs vfs, int nByte, string zErrMsg);
        public abstract IntPtr xDlSym(sqlite3_vfs vfs, IntPtr data, string zSymbol);
        public abstract int xDlClose(sqlite3_vfs vfs, IntPtr data);
        public abstract int xRandomness(sqlite3_vfs vfs, int nByte, byte[] buffer);
        public abstract int xSleep(sqlite3_vfs vfs, int microseconds);
        public abstract int xCurrentTime(sqlite3_vfs vfs, ref double currenttime);
        public abstract int xGetLastError(sqlite3_vfs pVfs, int nBuf, ref string zBuf);
        public abstract int xCurrentTimeInt64(sqlite3_vfs pVfs, ref long pTime);
        public abstract int xSetSystemCall(sqlite3_vfs pVfs, string zName, long sqlite3_syscall_ptr);
        public abstract int xGetSystemCall(sqlite3_vfs pVfs, string zName, long sqlite3_syscall_ptr);
        public abstract int xNextSystemCall(sqlite3_vfs pVfs, string zName, long sqlite3_syscall_ptr);

        public sqlite3_vfs() { }
        public sqlite3_vfs(int iVersion, int szOsFile, int mxPathname, sqlite3_vfs pNext, string zName, object pAppData)
        {
            this.iVersion = iVersion;
            this.szOsFile = szOsFile;
            this.mxPathname = mxPathname;
            this.pNext = pNext;
            this.zName = zName;
            this.pAppData = pAppData;
        }

        public void CopyTo(sqlite3_vfs ct)
        {
            ct.iVersion = this.iVersion;
            ct.szOsFile = this.szOsFile;
            ct.mxPathname = this.mxPathname;
            ct.pNext = this.pNext;
            ct.zName = this.zName;
            ct.pAppData = this.pAppData;
        }
    }
}
