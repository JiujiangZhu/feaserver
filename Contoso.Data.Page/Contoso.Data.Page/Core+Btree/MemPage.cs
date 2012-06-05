using System;
using Contoso.Sys;
using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;

namespace Contoso.Core
{
    public partial class MemPage
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
                    cp.pCell = MallocEx.sqlite3Malloc(pCell.Length);
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
        public DbPage pDbPage;          // Pager page handle
        public Pgno pgno;               // Page number for this page

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
                cp.aData = MallocEx.sqlite3Malloc(aData.Length);
                Buffer.BlockCopy(aData, 0, cp.aData, 0, aData.Length);
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

        internal static uint PENDING_BYTE_PAGE(BtShared pBt) { return (uint)Pager.PAGER_MJ_PGNO(pBt.pPager); }
        internal static Pgno PTRMAP_PAGENO(BtShared pBt, Pgno pgno) { return pBt.ptrmapPageno(pgno); }
        internal static uint PTRMAP_PTROFFSET(uint pgptrmap, uint pgno) { return (5 * (pgno - pgptrmap - 1)); }
        internal static bool PTRMAP_ISPAGE(BtShared pBt, uint pgno) { return (PTRMAP_PAGENO((pBt), (pgno)) == (pgno)); }
    }
}
