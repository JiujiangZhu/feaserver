namespace Contoso.Core
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
