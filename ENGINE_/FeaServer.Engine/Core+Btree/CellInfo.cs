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
    public struct CellInfo
    {
        public int iCell;           // Offset to start of cell content -- Needed for C#
        public byte[] pCell;        // Pointer to the start of cell content
        public long nKey;           // The key for INTKEY tables, or number of bytes in key
        public uint nData;          // Number of bytes of data
        public uint nPayload;       // Total amount of payload
        public ushort nHeader;      // Size of the cell content header in bytes
        public ushort nLocal;       // Amount of payload held locally
        public ushort iOverflow;    // Offset to overflow page number.  Zero if no overflow
        public ushort nSize;        // Size of the cell content on the main b-tree page

        public bool Equals(CellInfo ci)
        {
            if (ci.iCell >= ci.pCell.Length || iCell >= this.pCell.Length) return false;
            if (ci.pCell[ci.iCell] != this.pCell[iCell]) return false;
            if (ci.nKey != this.nKey || ci.nData != this.nData || ci.nPayload != this.nPayload) return false;
            if (ci.nHeader != this.nHeader || ci.nLocal != this.nLocal) return false;
            if (ci.iOverflow != this.iOverflow || ci.nSize != this.nSize) return false;
            return true;
        }
    }
}
