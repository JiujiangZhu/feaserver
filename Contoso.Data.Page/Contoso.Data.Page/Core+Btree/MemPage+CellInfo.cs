namespace Contoso.Core
{
    public partial class MemPage
    {
        public struct CellInfo
        {
            public int CellID;           // Offset to start of cell content -- Needed for C#
            public byte[] Cells;        // Pointer to the start of cell content   ??payload??
            public long nKey;           // The key for INTKEY tables, or number of bytes in key
            public uint nData;          // Number of bytes of data
            public uint nPayload;       // Total amount of payload
            public ushort nHeader;      // Size of the cell content header in bytes
            public ushort nLocal;       // Amount of payload held locally
            public ushort iOverflow;    // Offset to overflow page number.  Zero if no overflow
            public ushort nSize;        // Size of the cell content on the main b-tree page

            public bool Equals(CellInfo ci)
            {
                if (ci.CellID >= ci.Cells.Length || CellID >= Cells.Length) return false;
                if (ci.Cells[ci.CellID] != Cells[CellID]) return false;
                if (ci.nKey != nKey || ci.nData != nData || ci.nPayload != this.nPayload) return false;
                if (ci.nHeader != nHeader || ci.nLocal != nLocal) return false;
                if (ci.iOverflow != iOverflow || ci.nSize != nSize) return false;
                return true;
            }
        }
    }
}
