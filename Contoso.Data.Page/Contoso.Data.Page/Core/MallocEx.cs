using System;
namespace Contoso.Core
{
    public class MallocEx
    {
        internal static void sqlite3BeginBenignMalloc() { }
        internal static void sqlite3EndBenignMalloc() { }
        internal static void sqlite3_free(ref byte[] p) { p = null; }
        internal static byte[] sqlite3Malloc(int p) { return new byte[p]; }
        internal static bool sqlite3HeapNearlyFull() { return false; }
        internal static int sqlite3MallocSize(byte[] p) { return p.Length; }

        internal static byte[][] sqlite3ScratchMalloc(byte[][] apCell, int nMaxCells)
        {
            throw new NotImplementedException();
        }

        internal static void sqlite3ScratchFree(byte[][] apCell)
        {
        }
    }
}
