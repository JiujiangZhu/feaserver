namespace Contoso.IO
{
    public partial class VirtualFile
    {
        // Read a 32-bit integer from the given file descriptor.  Store the integer that is read in pRes.  Return SQLITE.OK if everything worked, or an
        // error code is something goes wrong.
        // All values are stored on disk as big-endian.
        public RC ReadByte(int offset, ref int pRes)
        {
            uint u32_pRes = 0;
            var rc = ReadByte(offset, ref u32_pRes);
            pRes = (int)u32_pRes;
            return rc;
        }
        public RC ReadByte(long offset, ref uint pRes) { return ReadByte((int)offset, ref pRes); }
        public RC ReadByte(int offset, ref uint pRes)
        {
            var ac = new byte[4];
            var rc = this.Read(ac, ac.Length, offset);
            pRes = (rc == RC.OK ? ConvertEx.Get4(ac) : 0);
            return rc;
        }

        // Write a 32-bit integer into the given file descriptor.  Return SQLITE.OK on success or an error code is something goes wrong.
        public RC WriteByte(long offset, uint val)
        {
            var ac = new byte[4];
            ConvertEx.Put4(ac, val);
            return this.Write(ac, 4, offset);
        }
    }
}