using System;
using System.Diagnostics;
namespace Contoso.Collections
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

        public Bitvec(uint iSize)
        {
            this.iSize = iSize;
        }

        public static void sqlite3BitvecDestroy(ref Bitvec p)
        {
            if (p == null)
                return;
            if (p.iDivisor != 0)
                for (uint i = 0; i < BITVEC_NPTR; i++)
                    sqlite3BitvecDestroy(ref p.u.apSub[i]);
        }

        public uint sqlite3BitvecSize() { return iSize; }
        public static implicit operator bool(Bitvec b) { return (b != null); }

        public int sqlite3BitvecTest(uint i)
        {
            if (i == 0 || i > iSize)
                return 0;
            i--;
            var p = this;
            while (iDivisor != 0)
            {
                uint bin = i / p.iDivisor;
                i %= p.iDivisor;
                p = p.u.apSub[bin];
                if (p == null)
                    return 0;
            }
            if (p.iSize <= BITVEC_NBIT)
                return (((p.u.aBitmap[i / BITVEC_SZELEM] & (1 << (int)(i & (BITVEC_SZELEM - 1)))) != 0) ? 1 : 0);
            uint h = BITVEC_HASH(i++);
            while (p.u.aHash[h] != 0)
            {
                if (p.u.aHash[h] == i)
                    return 1;
                h = (h + 1) % BITVEC_NINT;
            }
            return 0;
        }

        public RC sqlite3BitvecSet(uint i)
        {
            Debug.Assert(i > 0);
            Debug.Assert(i <= iSize);
            i--;
            var p = this;
            while (iSize > BITVEC_NBIT && iDivisor != 0)
            {
                uint bin = i / p.iDivisor;
                i %= p.iDivisor;
                if (p.u.apSub[bin] == null)
                    p.u.apSub[bin] = new Bitvec(p.iDivisor);
                p = p.u.apSub[bin];
            }
            if (p.iSize <= BITVEC_NBIT)
            {
                p.u.aBitmap[i / BITVEC_SZELEM] |= (byte)(1 << (int)(i & (BITVEC_SZELEM - 1)));
                return RC.OK;
            }
            var h = BITVEC_HASH(i++);
            // if there wasn't a hash collision, and this doesn't completely fill the hash, then just add it without worring about sub-dividing and re-hashing.
            if (p.u.aHash[h] == 0)
                if (p.nSet < (BITVEC_NINT - 1))
                    goto bitvec_set_end;
                else
                    goto bitvec_set_rehash;
            // there was a collision, check to see if it's already in hash, if not, try to find a spot for it 
            do
            {
                if (p.u.aHash[h] == i)
                    return RC.OK;
                h++;
                if (h >= BITVEC_NINT)
                    h = 0;
            } while (p.u.aHash[h] != 0);
        // we didn't find it in the hash.  h points to the first available free spot. check to see if this is going to make our hash too "full".
        bitvec_set_rehash:
            if (p.nSet >= BITVEC_MXHASH)
            {
                var aiValues = new uint[BITVEC_NINT];
                Buffer.BlockCopy(p.u.aHash, 0, aiValues, 0, aiValues.Length * (sizeof(uint)));
                p.u.apSub = new Bitvec[BITVEC_NPTR];
                p.iDivisor = (uint)((p.iSize + BITVEC_NPTR - 1) / BITVEC_NPTR);
                var rc = p.sqlite3BitvecSet(i);
                for (uint j = 0; j < BITVEC_NINT; j++)
                    if (aiValues[j] != 0)
                        rc |= p.sqlite3BitvecSet(aiValues[j]);
                return rc;
            }
        bitvec_set_end:
            p.nSet++;
            p.u.aHash[h] = i;
            return RC.OK;
        }

        public void sqlite3BitvecClear(uint i, uint[] pBuf)
        {
            Debug.Assert(i > 0);
            i--;
            var p = this;
            while (p.iDivisor != 0)
            {
                uint bin = i / p.iDivisor;
                i %= p.iDivisor;
                p = p.u.apSub[bin];
                if (null == p)
                    return;
            }
            if (p.iSize <= BITVEC_NBIT)
                p.u.aBitmap[i / BITVEC_SZELEM] &= (byte)~((1 << (int)(i & (BITVEC_SZELEM - 1))));
            else
            {
                var aiValues = pBuf;
                Array.Copy(p.u.aHash, aiValues, p.u.aHash.Length);
                p.u.aHash = new uint[aiValues.Length];
                p.nSet = 0;
                for (uint j = 0; j < BITVEC_NINT; j++)
                    if (aiValues[j] != 0 && aiValues[j] != (i + 1))
                    {
                        var h = BITVEC_HASH(aiValues[j] - 1);
                        p.nSet++;
                        while (p.u.aHash[h] != 0)
                        {
                            h++;
                            if (h >= BITVEC_NINT)
                                h = 0;
                        }
                        p.u.aHash[h] = aiValues[j];
                    }
            }
        }
    }
}
