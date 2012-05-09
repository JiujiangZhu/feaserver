using System;
using Pgno = System.UInt32;
using Contoso.Sys;
namespace Contoso.Core
{
    public partial class Pager
    {
        // Read a 32-bit integer from the given file descriptor.  Store the integer that is read in pRes.  Return SQLITE_OK if everything worked, or an
        // error code is something goes wrong.
        // All values are stored on disk as big-endian.
        static SQLITE read32bits(sqlite3_file fd, int offset, ref int pRes)
        {
            uint u32_pRes = 0;
            var rc = read32bits(fd, offset, ref u32_pRes);
            pRes = (int)u32_pRes;
            return rc;
        }
        static SQLITE read32bits(sqlite3_file fd, long offset, ref uint pRes) { return read32bits(fd, (int)offset, ref pRes); }
        static SQLITE read32bits(sqlite3_file fd, int offset, ref uint pRes)
        {
            var ac = new byte[4];
            var rc = sqlite3OsRead(fd, ac, ac.Length, offset);
            pRes = (rc == SQLITE.OK ? sqlite3Get4byte(ac) : 0);
            return rc;
        }

        // Write a 32-bit integer into a string buffer in big-endian byte order.
        static void put32bits(string ac, int offset, int val)
        {
            var A = new byte[4];
            A[0] = (byte)ac[offset + 0];
            A[1] = (byte)ac[offset + 1];
            A[2] = (byte)ac[offset + 2];
            A[3] = (byte)ac[offset + 3];
            sqlite3Put4byte(A, 0, val);
        }
        static void put32bits(byte[] ac, int offset, int val) { sqlite3Put4byte(ac, offset, (uint)val); }
        static void put32bits(byte[] ac, uint val) { sqlite3Put4byte(ac, 0U, val); }
        static void put32bits(byte[] ac, int offset, uint val) { sqlite3Put4byte(ac, offset, val); }

        // Write a 32-bit integer into the given file descriptor.  Return SQLITE_OK on success or an error code is something goes wrong.
        static SQLITE write32bits(sqlite3_file fd, long offset, uint val)
        {
            var ac = new byte[4];
            put32bits(ac, val);
            return sqlite3OsWrite(fd, ac, 4, offset);
        }
    }
}
