﻿#region License
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
using System;
namespace FeaServer.Engine.Core
{
    public class MemPage
    {
        public struct _OvflCell
        {   // Cells that will not fit on aData[]
            public byte[] pCell;        // Pointers to the body of the overflow cell
            public ushort idx;          // Insert this cell before idx-th non-overflow cell
            public _OvflCell Copy()
            {
                var cp = new _OvflCell();
                if (pCell != null)
                {
                    cp.pCell = new byte[pCell.Length];
                    Buffer.BlockCopy(pCell, 0, cp.pCell, 0, pCell.Length);
                }
                cp.idx = idx;
                return cp;
            }
        }

        public byte isInit;             // True if previously initialized. MUST BE FIRST!
        public byte nOverflow;          // Number of overflow cell bodies in aCell[]
        public byte intKey;             // True if u8key flag is set
        public byte leaf;               // 1 if leaf flag is set
        public byte hasData;            // True if this page stores data
        public byte hdrOffset;          // 100 for page 1.  0 otherwise
        public byte childPtrSize;       // 0 if leaf==1.  4 if leaf==0
        public ushort maxLocal;         // Copy of BtShared.maxLocal or BtShared.maxLeaf
        public ushort minLocal;         // Copy of BtShared.minLocal or BtShared.minLeaf
        public ushort cellOffset;       // Index in aData of first cell pou16er
        public ushort nFree;            // Number of free bytes on the page
        public ushort nCell;            // Number of cells on this page, local and ovfl
        public ushort maskPage;         // Mask for page offset
        public _OvflCell[] aOvfl = new _OvflCell[5];
        public BtShared pBt;            // Pointer to BtShared that this page is part of
        public byte[] aData;            // Pointer to disk image of the page data
        public PgHdr pDbPage;           // Pager page handle
        public uint pgno;               // Page number for this page

        public MemPage Copy()
        {
            var cp = (MemPage)MemberwiseClone();
            if (aOvfl != null)
            {
                cp.aOvfl = new _OvflCell[aOvfl.Length];
                for (var i = 0; i < aOvfl.Length; i++)
                    cp.aOvfl[i] = aOvfl[i].Copy();
            }
            if (aData != null)
            {
                cp.aData = new byte[aData.Length];
                Buffer.BlockCopy(aData, 0, cp.aData, 0, aData.Length);
            }
            return cp;
        }
    }
}
