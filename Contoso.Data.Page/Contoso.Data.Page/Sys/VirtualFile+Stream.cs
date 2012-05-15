using Contoso.Core;
namespace Contoso.Sys
{
    public partial class VirtualFile
    {
        // Read a 32-bit integer from the given file descriptor.  Store the integer that is read in pRes.  Return SQLITE.OK if everything worked, or an
        // error code is something goes wrong.
        // All values are stored on disk as big-endian.
        internal SQLITE read32bits(int offset, ref int pRes)
        {
            uint u32_pRes = 0;
            var rc = read32bits(offset, ref u32_pRes);
            pRes = (int)u32_pRes;
            return rc;
        }
        internal SQLITE read32bits(long offset, ref uint pRes) { return read32bits((int)offset, ref pRes); }
        internal SQLITE read32bits(int offset, ref uint pRes)
        {
            var ac = new byte[4];
            var rc = FileEx.sqlite3OsRead(this, ac, ac.Length, offset);
            pRes = (rc == SQLITE.OK ? ConvertEx.sqlite3Get4byte(ac) : 0);
            return rc;
        }

        // Write a 32-bit integer into the given file descriptor.  Return SQLITE.OK on success or an error code is something goes wrong.
        internal SQLITE write32bits(long offset, uint val)
        {
            var ac = new byte[4];
            ConvertEx.put32bits(ac, val);
            return FileEx.sqlite3OsWrite(this, ac, 4, offset);
        }
    }
}