using System;
namespace Contoso.Core
{
    public class MallocEx
    {
        internal static void sqlite3BeginBenignMalloc()
        {
        }

        internal static void sqlite3EndBenignMalloc()
        {
        }

        internal static void sqlite3_free(ref byte[] p) { p = null; }
        internal static byte[] sqlite3Malloc(int p) { return new byte[p]; }

        internal static bool sqlite3HeapNearlyFull()
        {
            return false;
        }

        internal static int sqlite3MallocSize(byte[] p)
        {
            throw new NotImplementedException();
        }

    }
}
