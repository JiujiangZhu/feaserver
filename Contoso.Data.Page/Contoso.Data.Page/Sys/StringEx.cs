using System;
using System.Runtime.InteropServices;
using System.Text;
namespace Contoso.Sys
{
    public static class StringEx
    {
        public static int sqlite3Strlen30(int z) { return 0x3fffffff & z;  }
        public static int sqlite3Strlen30(StringBuilder z)
        {
            if (z == null)
                return 0;
            var iLen = z.ToString().IndexOf('\0');
            return 0x3fffffff & (iLen == -1 ? z.Length : iLen);
        }
        public static int sqlite3Strlen30(string z)
        {
            if (z == null)
                return 0;
            var iLen = z.IndexOf('\0');
            return 0x3fffffff & (iLen == -1 ? z.Length : iLen);
        }
    }
}
