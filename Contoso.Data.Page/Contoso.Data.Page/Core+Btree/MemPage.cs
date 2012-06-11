using System;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class MemPage
    {
        // Cells that will not fit on aData[]
        public struct Overflow
        {
            public byte[] Cell;     // Pointers to the body of the overflow cell
            public ushort Index;    // Insert this cell before idx-th non-overflow cell
            public Overflow Clone()
            {
                var cp = new Overflow();
                if (Cell != null)
                {
                    cp.Cell = MallocEx.sqlite3Malloc(Cell.Length);
                    Buffer.BlockCopy(Cell, 0, cp.Cell, 0, Cell.Length);
                }
                cp.Index = Index;
                return cp;
            }
        }

        public bool HasInit;            // True if previously initialized. MUST BE FIRST!
        public byte NOverflows;          // Number of overflow cell bodies in aCell[]
        public bool HasIntKey;          // True if u8key flag is set
        public byte Leaf;               // 1 if leaf flag is set
        public byte HasData;            // True if this page stores data
        public byte HeaderOffset;       // 100 for page 1.  0 otherwise
        public byte ChildPtrSize;       // 0 if leaf==1.  4 if leaf==0
        public ushort MaxLocal;         // Copy of BtShared.maxLocal or BtShared.maxLeaf
        public ushort MinLocal;         // Copy of BtShared.minLocal or BtShared.minLeaf
        public ushort CellOffset;       // Index in aData of first cell pou16er
        public ushort FreeBytes;        // Number of free bytes on the page
        public ushort Cells;            // Number of cells on this page, local and ovfl
        public ushort MaskPage;         // Mask for page offset
        public Overflow[] Overflows = new Overflow[5];
        public BtShared Shared;         // Pointer to BtShared that this page is part of
        public byte[] Data;             // Pointer to disk image of the page data
        public DbPage DbPage;           // Pager page handle
        // was:pgno
        public Pgno ID;                 // Page number for this page

        public MemPage Clone()
        {
            var cp = (MemPage)MemberwiseClone();
            if (Overflows != null)
            {
                cp.Overflows = new Overflow[Overflows.Length];
                for (var i = 0; i < Overflows.Length; i++)
                    cp.Overflows[i] = Overflows[i].Clone();
            }
            if (Data != null)
            {
                cp.Data = MallocEx.sqlite3Malloc(Data.Length);
                Buffer.BlockCopy(Data, 0, cp.Data, 0, Data.Length);
            }
            return cp;
        }

        public enum PTRMAP : byte
        {
            ROOTPAGE = 1,
            FREEPAGE = 2,
            OVERFLOW1 = 3,
            OVERFLOW2 = 4,
            BTREE = 5,
        }

        internal static uint PENDING_BYTE_PAGE(BtShared pBt) { return (uint)Pager.PAGER_MJ_PGNO(pBt.Pager); }
        internal static Pgno PTRMAP_PAGENO(BtShared pBt, Pgno pgno) { return pBt.ptrmapPageno(pgno); }
        internal static uint PTRMAP_PTROFFSET(uint pgptrmap, uint pgno) { return (5 * (pgno - pgptrmap - 1)); }
        internal static bool PTRMAP_ISPAGE(BtShared pBt, uint pgno) { return (PTRMAP_PAGENO((pBt), (pgno)) == (pgno)); }
    }
}
