using System;
using System.Diagnostics;
namespace Contoso.Collections
{
    public class BitArray
    {
        private static int BITVEC_SZ = 512;
        private static int BITVEC_USIZE = (((BITVEC_SZ - (3 * sizeof(uint))) / 4) * 4);
        private const int BITVEC_SZELEM = 8;
        private static int BITVEC_NELEM = (int)(BITVEC_USIZE / sizeof(byte));
        private static int BITVEC_NBIT = (BITVEC_NELEM * BITVEC_SZELEM);
        private static uint BITVEC_NINT = (uint)(BITVEC_USIZE / sizeof(uint));
        private static int BITVEC_MXHASH = (int)(BITVEC_NINT / 2);
        private static uint BITVEC_HASH(uint X) { return (uint)(((X) * 1) % BITVEC_NINT); }
        private static int BITVEC_NPTR = (int)(BITVEC_USIZE / 4);

        private uint iSize;      // Maximum bit index.  Max iSize is 4,294,967,296.
        private uint nSet;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
        private uint iDivisor;   // Number of bits handled by each apSub[] entry.
        // Should >=0 for apSub element.
        // Max iDivisor is max(u32) / BITVEC_NPTR + 1.
        // For a BITVEC_SZ of 512, this would be 34,359,739.
        private class _u
        {
            public byte[] aBitmap = new byte[BITVEC_NELEM]; // Bitmap representation
            public uint[] aHash = new uint[BITVEC_NINT];              // Hash table representation
            public BitArray[] apSub = new BitArray[BITVEC_NPTR];        // Recursive representation
        }
        private _u u = new _u();

        public BitArray(uint iSize)
        {
            this.iSize = iSize;
        }

        public static void Destroy(ref BitArray p)
        {
            if (p == null)
                return;
            if (p.iDivisor != 0)
                for (uint i = 0; i < BITVEC_NPTR; i++)
                    Destroy(ref p.u.apSub[i]);
        }

        public uint Length { get { return iSize; } }

        public static implicit operator bool(BitArray b) { return (b != null); }

        public int Get(uint index)
        {
            if (index == 0 || index > iSize)
                return 0;
            index--;
            var p = this;
            while (iDivisor != 0)
            {
                uint bin = index / p.iDivisor;
                index %= p.iDivisor;
                p = p.u.apSub[bin];
                if (p == null)
                    return 0;
            }
            if (p.iSize <= BITVEC_NBIT)
                return (((p.u.aBitmap[index / BITVEC_SZELEM] & (1 << (int)(index & (BITVEC_SZELEM - 1)))) != 0) ? 1 : 0);
            uint h = BITVEC_HASH(index++);
            while (p.u.aHash[h] != 0)
            {
                if (p.u.aHash[h] == index)
                    return 1;
                h = (h + 1) % BITVEC_NINT;
            }
            return 0;
        }

        public RC Set(uint index)
        {
            Debug.Assert(index > 0);
            Debug.Assert(index <= iSize);
            index--;
            var p = this;
            while (iSize > BITVEC_NBIT && iDivisor != 0)
            {
                uint bin = index / p.iDivisor;
                index %= p.iDivisor;
                if (p.u.apSub[bin] == null)
                    p.u.apSub[bin] = new BitArray(p.iDivisor);
                p = p.u.apSub[bin];
            }
            if (p.iSize <= BITVEC_NBIT)
            {
                p.u.aBitmap[index / BITVEC_SZELEM] |= (byte)(1 << (int)(index & (BITVEC_SZELEM - 1)));
                return RC.OK;
            }
            var h = BITVEC_HASH(index++);
            // if there wasn't a hash collision, and this doesn't completely fill the hash, then just add it without worring about sub-dividing and re-hashing.
            if (p.u.aHash[h] == 0)
                if (p.nSet < (BITVEC_NINT - 1))
                    goto bitvec_set_end;
                else
                    goto bitvec_set_rehash;
            // there was a collision, check to see if it's already in hash, if not, try to find a spot for it 
            do
            {
                if (p.u.aHash[h] == index)
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
                p.u.apSub = new BitArray[BITVEC_NPTR];
                p.iDivisor = (uint)((p.iSize + BITVEC_NPTR - 1) / BITVEC_NPTR);
                var rc = p.Set(index);
                for (uint j = 0; j < BITVEC_NINT; j++)
                    if (aiValues[j] != 0)
                        rc |= p.Set(aiValues[j]);
                return rc;
            }
        bitvec_set_end:
            p.nSet++;
            p.u.aHash[h] = index;
            return RC.OK;
        }

        public void Clear(uint index, uint[] buffer)
        {
            Debug.Assert(index > 0);
            index--;
            var p = this;
            while (p.iDivisor != 0)
            {
                uint bin = index / p.iDivisor;
                index %= p.iDivisor;
                p = p.u.apSub[bin];
                if (null == p)
                    return;
            }
            if (p.iSize <= BITVEC_NBIT)
                p.u.aBitmap[index / BITVEC_SZELEM] &= (byte)~((1 << (int)(index & (BITVEC_SZELEM - 1))));
            else
            {
                var aiValues = buffer;
                Array.Copy(p.u.aHash, aiValues, p.u.aHash.Length);
                p.u.aHash = new uint[aiValues.Length];
                p.nSet = 0;
                for (uint j = 0; j < BITVEC_NINT; j++)
                    if (aiValues[j] != 0 && aiValues[j] != (index + 1))
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
