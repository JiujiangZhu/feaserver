#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
namespace FeaServer.Engine.Core
{
    public class Bitvec
    {
        static int BITVEC_SZ = 512;
        static int BITVEC_USIZE = (((BITVEC_SZ - (3 * sizeof(uint))) / 4) * 4);
        const int BITVEC_SZELEM = 8;
        static int BITVEC_NELEM = (int)(BITVEC_USIZE / sizeof(byte));
        static int BITVEC_NBIT = (BITVEC_NELEM * BITVEC_SZELEM);
        static uint BITVEC_NINT = (uint)(BITVEC_USIZE / sizeof(uint));
        static int BITVEC_MXHASH = (int)(BITVEC_NINT / 2);
        static uint BITVEC_HASH(uint X) { return (uint)(((X) * 1) % BITVEC_NINT); }
        static int BITVEC_NPTR = (int)(BITVEC_USIZE / 4);

        public uint iSize;      // Maximum bit index.  Max iSize is 4,294,967,296.
        public uint nSet;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
        public uint iDivisor;   // Number of bits handled by each apSub[] entry.
        // Should >=0 for apSub element.
        // Max iDivisor is max(u32) / BITVEC_NPTR + 1.
        // For a BITVEC_SZ of 512, this would be 34,359,739.
        public class _u
        {
            public byte[] aBitmap = new byte[BITVEC_NELEM]; // Bitmap representation
            public uint[] aHash = new uint[BITVEC_NINT];              // Hash table representation
            public Bitvec[] apSub = new Bitvec[BITVEC_NPTR];        // Recursive representation
        }
        public _u u = new _u();

        public static implicit operator bool(Bitvec b)
        {
            return (b != null);
        }
    }
}
