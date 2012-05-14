using System;
using System.Runtime.InteropServices;
namespace Contoso.Sys
{
    public static class ArrayEx
    {
        //[DllImport("msvcrt.dll")]
        //private static extern unsafe int memcmp(byte* a, byte* b, int count);

        //public static unsafe int Compare(byte[] a, byte[] b, int count)
        //{
        //    fixed (byte* a1 = a, b1 = b)
        //        return memcmp(a1, b1, count);
        //}

        public static int Compare(byte[] a, byte[] b, int count)
        {
            if (a.Length < count) return (a.Length < b.Length ? -1 : 1);
            if (b.Length < count) return 1;
            for (int i = 0; i < count; i++)
                if (a[i] != b[i]) return (a[i] < b[i] ? -1 : 1);
            return 0;
        }

        //public static unsafe int Compare(byte[] buffer1, int offset1, byte[] buffer2, int offset2, int count)
        //{
        //    fixed (byte* b1 = buffer1, b2 = buffer2)
        //        return memcmp(b1 + offset1, b2 + offset2, count);
        //}
    }
}
