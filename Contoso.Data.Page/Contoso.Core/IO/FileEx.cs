using System.Diagnostics;
namespace Contoso.IO
{
    public partial class FileEx
    {
        public static RC sqlite3OsClose(VirtualFile id)
        {
            var rc = RC.OK;
            if (id.IsOpen) { rc = id.xClose(); id.IsOpen = false; }
            return rc;
        }
        public static RC sqlite3OsCloseFree(VirtualFile id)
        {
            var rc = RC.OK;
            Debug.Assert(id != null);
            rc = sqlite3OsClose(id);
            return rc;
        }
        //internal static SQLITE sqlite3OsRead(VirtualFile id, byte[] pBuf, int amt, long offset) { if (pBuf == null) pBuf = MallocEx.sqlite3Malloc(amt); return id.xRead(pBuf, amt, offset); }
        //internal static SQLITE sqlite3OsWrite(VirtualFile id, byte[] pBuf, int amt, long offset) { return id.xWrite(pBuf, amt, offset); }
        //internal static SQLITE sqlite3OsTruncate(VirtualFile id, long size) { return id.xTruncate(size); }
        //internal static SQLITE sqlite3OsSync(VirtualFile id, VirtualFile.SYNC flags) { return id.xSync(flags); }
        //internal static SQLITE sqlite3OsFileSize(VirtualFile id, ref long pSize) { return id.xFileSize(ref pSize); }
        //internal static SQLITE sqlite3OsLock(VirtualFile id, VirtualFile.LOCK lockType) { return id.xLock(lockType); }
        //internal static SQLITE sqlite3OsUnlock(VirtualFile id, VirtualFile.LOCK lockType) { return id.xUnlock(lockType); }
        //internal static SQLITE sqlite3OsCheckReservedLock(VirtualFile id, ref int pResOut) { return id.xCheckReservedLock(ref pResOut); }
        //internal static SQLITE sqlite3OsFileControl(VirtualFile id, VirtualFile.FCNTL op, ref long pArg) { return id.xFileControl(op, ref pArg); }
        //internal static int sqlite3OsSectorSize(VirtualFile id) { return id.xSectorSize(); }
        //internal static VirtualFile.IOCAP sqlite3OsDeviceCharacteristics(VirtualFile id) { return id.xDeviceCharacteristics(); }
        //internal static SQLITE sqlite3OsShmLock(VirtualFile id, int offset, int n, int flags) { return id.xShmLock(offset, n, flags); }
        //internal static void sqlite3OsShmBarrier(VirtualFile id) { id.xShmBarrier(); }
        //internal static SQLITE sqlite3OsShmUnmap(VirtualFile id, int deleteFlag) { return id.xShmUnmap(deleteFlag); }
        //internal static SQLITE sqlite3OsShmMap(VirtualFile id, int iPage, int pgsz, int bExtend, out object pp) { return id.xShmMap(iPage, pgsz, bExtend, out pp); }
    }
}
