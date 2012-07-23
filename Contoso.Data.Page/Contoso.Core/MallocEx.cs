using System;
namespace Contoso
{
    public class MallocEx
    {
        public static void sqlite3BeginBenignMalloc() { }
        public static void sqlite3EndBenignMalloc() { }
        public static void sqlite3_free(ref byte[] p) { p = null; }
        public static byte[] sqlite3Malloc(int p) { return new byte[p]; }
        public static bool sqlite3HeapNearlyFull() { return false; }
        public static int sqlite3MallocSize(byte[] p) { return p.Length; }

        public static byte[][] sqlite3ScratchMalloc(byte[][] apCell, int nMaxCells)
        {
            throw new NotImplementedException();
        }

        public static void sqlite3ScratchFree(byte[][] apCell)
        {
        }
    }
}
