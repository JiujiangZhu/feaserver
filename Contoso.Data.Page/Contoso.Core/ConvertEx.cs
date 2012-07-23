using System.Diagnostics;
namespace Contoso
{
    public static class ConvertEx
    {
        private const int SLOT_2_0 = 0x001fc07f;
        private const uint SLOT_4_2_0 = (uint)0xf01fc07f;
        private const uint SQLITE_MAX_U32 = (uint)((((ulong)1) << 32) - 1);


        // The variable-length integer encoding is as follows:
        //
        // KEY:
        //         A = 0xxxxxxx    7 bits of data and one flag bit
        //         B = 1xxxxxxx    7 bits of data and one flag bit
        //         C = xxxxxxxx    8 bits of data
        //  7 bits - A
        // 14 bits - BA
        // 21 bits - BBA
        // 28 bits - BBBA
        // 35 bits - BBBBA
        // 42 bits - BBBBBA
        // 49 bits - BBBBBBA
        // 56 bits - BBBBBBBA
        // 64 bits - BBBBBBBBC
        public static byte GetVarint9(byte[] p, out int v) { v = p[0]; if (v <= 0x7F) return 1; ulong uv; var r = _GetVarint9L(p, 0, out uv); v = (int)uv; return r; }
        public static byte GetVarint9(byte[] p, out uint v) { v = p[0]; if (v <= 0x7F) return 1; ulong uv; var r = _GetVarint9L(p, 0, out uv); v = (uint)uv; return r; }
        public static byte GetVarint9(byte[] p, uint offset, out int v) { v = p[offset]; if (v <= 0x7F) return 1; ulong uv; var r = _GetVarint9L(p, offset, out uv); v = (int)uv; return r; }
        public static byte GetVarint9(byte[] p, uint offset, out uint v) { v = p[offset]; if (v <= 0x7F) return 1; ulong uv; var r = _GetVarint9L(p, offset, out uv); v = (uint)uv; return r; }
        public static byte GetVarint9L(byte[] p, uint offset, out long v) { v = p[offset]; if (v <= 0x7F) return 1; ulong uv; var r = _GetVarint9L(p, offset, out uv); v = (long)uv; return r; }
        public static byte GetVarint9L(byte[] p, uint offset, out ulong v) { v = p[offset]; if (v <= 0x7F) return 1; var r = _GetVarint9L(p, offset, out v); return r; }
        private static byte _GetVarint9L(byte[] p, uint offset, out ulong v)
        {
            uint a, b, s;
            a = p[offset + 0];
            // a: p0 (unmasked) 
            if (0 == (a & 0x80)) { v = a; return 1; }
            b = p[offset + 1];
            // b: p1 (unmasked)
            if (0 == (b & 0x80))
            {
                a &= 0x7f;
                a = a << 7;
                a |= b;
                v = a;
                return 2;
            }
            // Verify that constants are precomputed correctly
            Debug.Assert(SLOT_2_0 == ((0x7f << 14) | (0x7f)));
            Debug.Assert(SLOT_4_2_0 == ((0xfU << 28) | (0x7f << 14) | (0x7f)));
            a = a << 14;
            a |= p[offset + 2];
            // a: p0<<14 | p2 (unmasked)
            if (0 == (a & 0x80))
            {
                a &= SLOT_2_0;
                b &= 0x7f;
                b = b << 7;
                a |= b;
                v = a;
                return 3;
            }
            // CSE1 from below
            a &= SLOT_2_0;
            b = b << 14;
            b |= p[offset + 3];
            // b: p1<<14 | p3 (unmasked)
            if (0 == (b & 0x80))
            {
                b &= SLOT_2_0;
                // moved CSE1 up
                a = a << 7;
                a |= b;
                v = a;
                return 4;
            }
            // a: p0<<14 | p2 (masked)
            // b: p1<<14 | p3 (unmasked)
            // 1:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked)
            // moved CSE1 up
            b &= SLOT_2_0;
            s = a;
            // s: p0<<14 | p2 (masked)
            a = a << 14;
            a |= p[offset + 4];
            // a: p0<<28 | p2<<14 | p4 (unmasked)
            if (0 == (a & 0x80))
            {
                b = b << 7;
                a |= b;
                s = s >> 18;
                v = ((ulong)s) << 32 | a;
                return 5;
            }
            // 2:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked)
            s = s << 7;
            s |= b;
            // s: p0<<21 | p1<<14 | p2<<7 | p3 (masked)
            b = b << 14;
            b |= p[offset + 5];
            // b: p1<<28 | p3<<14 | p5 (unmasked) 
            if (0 == (b & 0x80))
            {
                a &= SLOT_2_0;
                a = a << 7;
                a |= b;
                s = s >> 18;
                v = ((ulong)s) << 32 | a;
                return 6;
            }
            a = a << 14;
            a |= p[offset + 6];
            // a: p2<<28 | p4<<14 | p6 (unmasked)
            if (0 == (a & 0x80))
            {
                a &= SLOT_4_2_0;
                b &= SLOT_2_0;
                b = b << 7;
                a |= b;
                s = s >> 11;
                v = ((ulong)s) << 32 | a;
                return 7;
            }
            /* CSE2 from below */
            a &= SLOT_2_0;
            //p++;
            b = b << 14;
            b |= p[offset + 7];
            // b: p3<<28 | p5<<14 | p7 (unmasked)
            if (0 == (b & 0x80))
            {
                b &= SLOT_4_2_0;
                // moved CSE2 up
                a = a << 7;
                a |= b;
                s = s >> 4;
                v = ((ulong)s) << 32 | a;
                return 8;
            }
            a = a << 15;
            a |= p[offset + 8];
            // a: p4<<29 | p6<<15 | p8 (unmasked) 
            // moved CSE2 up
            b &= SLOT_2_0;
            b = b << 8;
            a |= b;
            s = s << 4;
            b = p[offset + 4];
            b &= 0x7f;
            b = b >> 3;
            s |= b;
            v = ((ulong)s) << 32 | a;
            return 9;
        }

        public static byte GetVarint4(byte[] p, out int v) { v = p[0]; if (v <= 0x7F) return 1; uint uv; var r = _GetVarint(p, 0, out uv); v = (int)uv; return r; }
        public static byte GetVarint4(byte[] p, out uint v) { v = p[0]; if (v <= 0x7F) return 1; return _GetVarint(p, 0, out v); }
        public static byte GetVarint4(byte[] p, uint offset, out int v) { v = p[offset]; if (v <= 0x7F) return 1; uint uv; var r = _GetVarint(p, offset, out uv); v = (int)uv; return r; }
        public static byte GetVarint4(byte[] p, uint offset, out uint v) { v = p[offset]; if (v <= 0x7F) return 1; return _GetVarint(p, offset, out v); }
        public static byte GetVarint4(string p, uint offset, out int v)
        {
            v = p[(int)offset]; if (v <= 0x7F) return 1;
            var a = new byte[4];
            a[0] = (byte)p[(int)offset + 0];
            a[1] = (byte)p[(int)offset + 1];
            a[2] = (byte)p[(int)offset + 2];
            a[3] = (byte)p[(int)offset + 3];
            uint uv; var r = _GetVarint(a, 0, out uv); v = (int)uv; return r;
        }
        public static byte GetVarint4(string p, uint offset, out uint v)
        {
            v = p[(int)offset]; if (v <= 0x7F) return 1;
            var a = new byte[4];
            a[0] = (byte)p[(int)offset + 0];
            a[1] = (byte)p[(int)offset + 1];
            a[2] = (byte)p[(int)offset + 2];
            a[3] = (byte)p[(int)offset + 3];
            return _GetVarint(a, 0, out v);
        }
        private static byte _GetVarint(byte[] p, uint offset, out uint v)
        {
            uint a, b;
            // The 1-byte case.  Overwhelmingly the most common.  Handled inline  by the getVarin32() macro
            a = p[offset + 0];
            // a: p0 (unmasked)
            // The 2-byte case
            b = (offset + 1 < p.Length ? p[offset + 1] : (uint)0);
            // b: p1 (unmasked)
            if (0 == (b & 0x80))
            {
                // Values between 128 and 16383
                a &= 0x7f;
                a = a << 7;
                v = a | b;
                return 2;
            }
            // The 3-byte case
            a = a << 14;
            a |= (offset + 2 < p.Length ? p[offset + 2] : (uint)0);
            // a: p0<<14 | p2 (unmasked)
            if (0 == (a & 0x80))
            {
                // Values between 16384 and 2097151
                a &= (0x7f << 14) | (0x7f);
                b &= 0x7f;
                b = b << 7;
                v = a | b;
                return 3;
            }
            // A 32-bit varint is used to store size information in btrees. Objects are rarely larger than 2MiB limit of a 3-byte varint.
            // A 3-byte varint is sufficient, for example, to record the size of a 1048569-byte BLOB or string.
            // We only unroll the first 1-, 2-, and 3- byte cases.  The very rare larger cases can be handled by the slower 64-bit varint
            // routine.
            {
                ulong ulong_v = 0;
                byte n = _GetVarint9L(p, offset, out ulong_v);
                Debug.Assert(n > 3 && n <= 9);
                v = ((ulong_v & SQLITE_MAX_U32) != ulong_v ? 0xffffffff : (uint)ulong_v);
                return n;
            }
        }

        public static int PutVarint9(byte[] p, uint offset, int v) { return PutVarint9L(p, offset, (ulong)v); }
        public static int PutVarint9L(byte[] p, uint offset, ulong v)
        {
            int i, j, n;
            if ((v & (((ulong)0xff000000) << 32)) != 0)
            {
                p[offset + 8] = (byte)v;
                v >>= 8;
                for (i = 7; i >= 0; i--)
                {
                    p[offset + i] = (byte)((v & 0x7f) | 0x80);
                    v >>= 7;
                }
                return 9;
            }
            n = 0;
            var bufByte10 = new byte[10];
            do
            {
                bufByte10[n++] = (byte)((v & 0x7f) | 0x80);
                v >>= 7;
            } while (v != 0);
            bufByte10[0] &= 0x7f;
            Debug.Assert(n <= 9);
            for (i = 0, j = n - 1; j >= 0; j--, i++)
                p[offset + i] = bufByte10[j];
            return n;
        }

        public static int PutVarint4(byte[] p, int v)
        {
            if ((v & ~0x7f) == 0) { p[0] = (byte)v; return 1; }
            if ((v & ~0x3fff) == 0) { p[0] = (byte)((v >> 7) | 0x80); p[1] = (byte)(v & 0x7f); return 2; }
            return PutVarint9(p, 0, v);
        }

        public static int PutVarint4(byte[] p, uint offset, int v)
        {
            if ((v & ~0x7f) == 0) { p[offset] = (byte)v; return 1; }
            if ((v & ~0x3fff) == 0) { p[offset] = (byte)((v >> 7) | 0x80); p[offset + 1] = (byte)(v & 0x7f); return 2; }
            return PutVarint9(p, offset, v);
        }

        public static int GetVarintLength(ulong v)
        {
            var i = 0;
            do { i++; v >>= 7; }
            while (v != 0 && Check.ALWAYS(i < 9));
            return i;
        }

        public static uint Get4(byte[] p, int p_offset, int offset) { offset += p_offset; return (offset + 3 > p.Length) ? 0 : (uint)((p[0 + offset] << 24) | (p[1 + offset] << 16) | (p[2 + offset] << 8) | p[3 + offset]); }
        public static uint Get4(byte[] p, int offset) { return (offset + 3 > p.Length) ? 0 : (uint)((p[0 + offset] << 24) | (p[1 + offset] << 16) | (p[2 + offset] << 8) | p[3 + offset]); }
        public static uint Get4(byte[] p, uint offset) { return (offset + 3 > p.Length) ? 0 : (uint)((p[0 + offset] << 24) | (p[1 + offset] << 16) | (p[2 + offset] << 8) | p[3 + offset]); }
        public static uint Get4(byte[] p) { return (uint)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]); }
        public static void Put4(byte[] p, int v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, uint v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, int offset, int v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, uint offset, int v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, int offset, uint v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4(byte[] p, uint offset, uint v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4L(byte[] p, int offset, ulong v)
        {
            p[0 + offset] = (byte)(v >> 24 & 0xFF);
            p[1 + offset] = (byte)(v >> 16 & 0xFF);
            p[2 + offset] = (byte)(v >> 8 & 0xFF);
            p[3 + offset] = (byte)(v & 0xFF);
        }
        public static void Put4L(byte[] p, ulong v)
        {
            p[0] = (byte)(v >> 24 & 0xFF);
            p[1] = (byte)(v >> 16 & 0xFF);
            p[2] = (byte)(v >> 8 & 0xFF);
            p[3] = (byte)(v & 0xFF);
        }
        public static void Put4(string p, int offset, int v)
        {
            var a = new byte[4];
            a[0] = (byte)p[offset + 0];
            a[1] = (byte)p[offset + 1];
            a[2] = (byte)p[offset + 2];
            a[3] = (byte)p[offset + 3];
            Put4(a, 0, v);
        }
        public static void Put4(string p, int offset, uint v)
        {
            var a = new byte[4];
            a[0] = (byte)p[offset + 0];
            a[1] = (byte)p[offset + 1];
            a[2] = (byte)p[offset + 2];
            a[3] = (byte)p[offset + 3];
            Put4(a, 0, v);
        }

        public static int Get2(byte[] p, int offset) { return p[offset + 0] << 8 | p[offset + 1]; }
        public static int Get2nz(byte[] X, int offset) { return (((((int)Get2(X, offset)) - 1) & 0xffff) + 1); }
        public static void Put2(byte[] pData, int Offset, uint v)
        {
            pData[Offset + 0] = (byte)(v >> 8);
            pData[Offset + 1] = (byte)v;
        }
        public static void Put2(byte[] pData, int Offset, int v)
        {
            pData[Offset + 0] = (byte)(v >> 8);
            pData[Offset + 1] = (byte)v;
        }
    }
}
