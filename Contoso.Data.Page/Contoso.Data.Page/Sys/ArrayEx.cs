using System;
using System.Runtime.InteropServices;
namespace Contoso.Sys
{
    public static class ArrayEx
    {
        [DllImport("msvcrt.dll")]
        private static extern unsafe int memcmp(byte* b1, byte* b2, int count);

        public static unsafe int Compare(byte[] buffer1, byte[] buffer2, int count)
        {
            fixed (byte* b1 = buffer1, b2 = buffer2)
                return memcmp(b1, b2, count);
        }

        //public static int memcmp(byte[] bA, byte[] bB, int Limit)
        //{
        //    if (bA.Length < Limit)
        //        return (bA.Length < bB.Length) ? -1 : +1;
        //    if (bB.Length < Limit)
        //        return +1;
        //    for (int i = 0; i < Limit; i++)
        //    {
        //        if (bA[i] != bB[i])
        //            return (bA[i] < bB[i]) ? -1 : 1;
        //    }
        //    return 0;
        //}

        public static unsafe int Compare(byte[] buffer1, int offset1, byte[] buffer2, int offset2, int count)
        {
            fixed (byte* b1 = buffer1, b2 = buffer2)
                return memcmp(b1 + offset1, b2 + offset2, count);
        }
    }
}
