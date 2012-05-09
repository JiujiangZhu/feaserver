using System.IO;
namespace Contoso.Sys
{
    public interface sqlite3_io_methods
    {
        int iVersion { get; }
        int xClose(sqlite3_file File_ID);
        int xRead(sqlite3_file File_ID, byte[] buffer, int amount, long offset);
        int xWrite(sqlite3_file File_ID, byte[] buffer, int amount, long offset);
        int xTruncate(sqlite3_file File_ID, long size);
        int xSync(sqlite3_file File_ID, int flags);
        int xFileSize(sqlite3_file File_ID, ref long size);
        int xLock(sqlite3_file File_ID, int locktype);
        int xUnlock(sqlite3_file File_ID, int locktype);
        int xCheckReservedLock(sqlite3_file File_ID, ref int pRes);
        int xFileControl(sqlite3_file File_ID, int op, ref long pArgs);
        int xSectorSize(sqlite3_file File_ID);
        int xDeviceCharacteristics(sqlite3_file File_ID);
        int xShmMap(sqlite3_file File_ID, int iPg, int pgsz, int pInt, out object pvolatile);
        int xShmLock(sqlite3_file File_ID, int offset, int n, int flags);
        void xShmBarrier(sqlite3_file File_ID);
        int xShmUnmap(sqlite3_file File_ID, int deleteFlag);
    }
}
