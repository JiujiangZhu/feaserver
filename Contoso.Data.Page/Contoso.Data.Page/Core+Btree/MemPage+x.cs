using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class MemPage
    {
        // was:findCell
        internal int FindCell(int cellID) { return ConvertEx.Get2(Data, CellOffset + (2 * cellID)); }

        // was:findCellv2
        private static byte[] FindCellv2(byte[] page, ushort cellID, ushort o, int i) { Debugger.Break(); return page; }

        // was:findOverflowCell
        private int FindOverflowCell(int cellID)
        {
            Debug.Assert(MutexEx.Held(Shared.Mutex));
            for (var i = NOverflows - 1; i >= 0; i--)
            {
                var overflow = Overflows[i];
                var k = overflow.Index;
                if (k <= cellID)
                {
                    if (k == cellID)
                        return -i - 1; // Negative Offset means overflow cells
                    cellID--;
                }
            }
            return FindCell(cellID);
        }

        // was:btreeParseCellPtr
        private void btreeParseCellPtr(int cellID, ref CellInfo info) { btreeParseCellPtr(Data, cellID, ref info); }
        internal void btreeParseCellPtr(byte[] cell, ref CellInfo info) { btreeParseCellPtr(cell, 0, ref info); }
        internal void btreeParseCellPtr(byte[] cell, int cellID, ref CellInfo info)
        {
            var nPayload = (uint)0;    // Number of bytes of cell payload
            Debug.Assert(MutexEx.Held(Shared.Mutex));
            if (info.Cells != cell)
                info.Cells = cell;
            info.CellID = cellID;
            Debug.Assert(Leaf == 0 || Leaf == 1);
            var n = (ushort)ChildPtrSize; // Number bytes in cell content header
            Debug.Assert(n == 4 - 4 * Leaf);
            if (HasIntKey)
            {
                if (HasData != 0)
                    n += (ushort)ConvertEx.GetVarint4(cell, (uint)(cellID + n), out nPayload);
                else
                    nPayload = 0;
                n += (ushort)ConvertEx.GetVarint9L(cell, (uint)(cellID + n), out info.nKey);
                info.nData = nPayload;
            }
            else
            {
                info.nData = 0;
                n += (ushort)ConvertEx.GetVarint4(cell, (uint)(cellID + n), out nPayload);
                info.nKey = nPayload;
            }
            info.nPayload = nPayload;
            info.nHeader = n;
            if (Check.LIKELY(nPayload <= this.MaxLocal))
            {
                // This is the (easy) common case where the entire payload fits on the local page.  No overflow is required.
                if ((info.nSize = (ushort)(n + nPayload)) < 4)
                    info.nSize = 4;
                info.nLocal = (ushort)nPayload;
                info.iOverflow = 0;
            }
            else
            {
                // If the payload will not fit completely on the local page, we have to decide how much to store locally and how much to spill onto
                // overflow pages.  The strategy is to minimize the amount of unused space on overflow pages while keeping the amount of local storage
                // in between minLocal and maxLocal.
                // Warning:  changing the way overflow payload is distributed in any way will result in an incompatible file format.
                var minLocal = (int)MinLocal; // Minimum amount of payload held locally
                var maxLocal = (int)MaxLocal;// Maximum amount of payload held locally
                var surplus = (int)(minLocal + (nPayload - minLocal) % (Shared.UsableSize - 4));// Overflow payload available for local storage
                info.nLocal = (surplus <= maxLocal ? (ushort)surplus : (ushort)minLocal);
                info.iOverflow = (ushort)(info.nLocal + n);
                info.nSize = (ushort)(info.iOverflow + 4);
            }
        }

        // was:parseCell/btreeParseCell
        internal void ParseCell(int cellID, ref CellInfo info) { btreeParseCellPtr(FindCell(cellID), ref info); }

        internal ushort cellSizePtr(int iCell)
        {
            var info = new CellInfo();
            var pCell = new byte[13];
            // Minimum Size = (2 bytes of Header  or (4) Child Pointer) + (maximum of) 9 bytes data
            if (iCell < 0)// Overflow Cell
                Buffer.BlockCopy(this.Overflows[-(iCell + 1)].Cell, 0, pCell, 0, pCell.Length < this.Overflows[-(iCell + 1)].Cell.Length ? pCell.Length : this.Overflows[-(iCell + 1)].Cell.Length);
            else if (iCell >= this.Data.Length + 1 - pCell.Length)
                Buffer.BlockCopy(this.Data, iCell, pCell, 0, this.Data.Length - iCell);
            else
                Buffer.BlockCopy(this.Data, iCell, pCell, 0, pCell.Length);
            btreeParseCellPtr(pCell, ref info);
            return info.nSize;
        }
        internal ushort cellSizePtr(byte[] pCell, int offset)
        {
            var info = new CellInfo();
            info.Cells = MallocEx.sqlite3Malloc(pCell.Length);
            Buffer.BlockCopy(pCell, offset, info.Cells, 0, pCell.Length - offset);
            btreeParseCellPtr(info.Cells, ref info);
            return info.nSize;
        }
        internal ushort cellSizePtr(byte[] pCell)
        {
            int _pIter = this.ChildPtrSize;
            uint nSize = 0;

#if DEBUG
            // The value returned by this function should always be the same as the (CellInfo.nSize) value found by doing a full parse of the
            // cell. If SQLITE_DEBUG is defined, an Debug.Assert() at the bottom of this function verifies that this invariant is not violated. */
            var debuginfo = new CellInfo();
            btreeParseCellPtr(pCell, ref debuginfo);
#else
            var debuginfo = new CellInfo();
#endif
            if (this.HasIntKey)
            {
                if (this.HasData != 0)
                    _pIter += ConvertEx.GetVarint4(pCell, out nSize);
                else
                    nSize = 0;
                // pIter now points at the 64-bit integer key value, a variable length integer. The following block moves pIter to point at the first byte
                // past the end of the key value. */
                var pEnd = _pIter + 9;
                while (((pCell[_pIter++]) & 0x80) != 0 && _pIter < pEnd) ;
            }
            else
                _pIter += ConvertEx.GetVarint4(pCell, (uint)_pIter, out nSize);
            if (nSize > this.MaxLocal)
            {
                var minLocal = (int)this.MinLocal;
                nSize = (ushort)(minLocal + (nSize - minLocal) % (this.Shared.UsableSize - 4));
                if (nSize > this.MaxLocal)
                    nSize = (ushort)minLocal;
                nSize += 4;
            }
            nSize += (uint)_pIter;
            // The minimum size of any cell is 4 bytes.
            if (nSize < 4)
                nSize = 4;
            Debug.Assert(nSize == debuginfo.nSize);
            return (ushort)nSize;
        }
#if DEBUG
        internal ushort cellSize(int iCell) { return cellSizePtr(FindCell(iCell)); }
#else
        internal int cellSize(int iCell) { return -1; }
#endif

#if !SQLITE_OMIT_AUTOVACUUM
        internal void ptrmapPutOvflPtr(int pCell, ref RC pRC)
        {
            if (pRC != 0)
                return;
            var info = new CellInfo();
            Debug.Assert(pCell != 0);
            btreeParseCellPtr(pCell, ref info);
            Debug.Assert((info.nData + (this.HasIntKey ? 0 : info.nKey)) == info.nPayload);
            if (info.iOverflow != 0)
            {
                Pgno ovfl = ConvertEx.Get4(this.Data, pCell, info.iOverflow);
                this.Shared.ptrmapPut(ovfl, PTRMAP.OVERFLOW1, this.ID, ref pRC);
            }
        }

        internal void ptrmapPutOvflPtr(byte[] pCell, ref RC pRC)
        {
            if (pRC != 0)
                return;
            var info = new CellInfo();
            Debug.Assert(pCell != null);
            btreeParseCellPtr(pCell, ref info);
            Debug.Assert((info.nData + (this.HasIntKey ? 0 : info.nKey)) == info.nPayload);
            if (info.iOverflow != 0)
            {
                Pgno ovfl = ConvertEx.Get4(pCell, info.iOverflow);
                this.Shared.ptrmapPut(ovfl, PTRMAP.OVERFLOW1, this.ID, ref pRC);
            }
        }
#endif

        internal RC defragmentPage()
        {
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            Debug.Assert(this.Shared != null);
            Debug.Assert(this.Shared.UsableSize <= Pager.SQLITE_MAX_PAGE_SIZE);
            Debug.Assert(this.NOverflows == 0);
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            var temp = this.Shared.Pager.sqlite3PagerTempSpace();// Temp area for cell content
            var data = this.Data; // The page data
            var hdr = this.HeaderOffset; // Offset to the page header
            var cellOffset = this.CellOffset;// Offset to the cell pointer array
            var nCell = this.Cells;// Number of cells on the page
            Debug.Assert(nCell == ConvertEx.Get2(data, hdr + 3));
            var usableSize = (int)this.Shared.UsableSize;// Number of usable bytes on a page
            var cbrk = (int)ConvertEx.Get2(data, hdr + 5);// Offset to the cell content area
            Buffer.BlockCopy(data, cbrk, temp, cbrk, usableSize - cbrk);
            cbrk = usableSize;
            var iCellFirst = cellOffset + 2 * nCell; // First allowable cell index
            var iCellLast = usableSize - 4; // Last possible cell index
            for (var i = 0; i < nCell; i++)
            {
                var pAddr = cellOffset + i * 2; // The i-th cell pointer
                var pc = ConvertEx.Get2(data, pAddr); // Address of a i-th cell 
#if !SQLITE_ENABLE_OVERSIZE_CELL_CHECK
                // These conditions have already been verified in btreeInitPage() if SQLITE_ENABLE_OVERSIZE_CELL_CHECK is defined
                if (pc < iCellFirst || pc > iCellLast)
                    return SysEx.SQLITE_CORRUPT_BKPT();
#endif
                Debug.Assert(pc >= iCellFirst && pc <= iCellLast);
                var size = cellSizePtr(temp, pc); // Size of a cell 
                cbrk -= size;
                if (cbrk < iCellFirst || pc + size > usableSize)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                Debug.Assert(cbrk + size <= usableSize && cbrk >= iCellFirst);
                Buffer.BlockCopy(temp, pc, data, cbrk, size);
                ConvertEx.Put2(data, pAddr, cbrk);
            }
            Debug.Assert(cbrk >= iCellFirst);
            ConvertEx.Put2(data, hdr + 5, cbrk);
            data[hdr + 1] = 0;
            data[hdr + 2] = 0;
            data[hdr + 7] = 0;
            var addr = cellOffset + 2 * nCell; // Offset of first byte after cell pointer array 
            Array.Clear(data, addr, cbrk - addr);
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            return (cbrk - iCellFirst != this.FreeBytes ? SysEx.SQLITE_CORRUPT_BKPT() : RC.OK);
        }

        internal RC allocateSpace(int nByte, ref int pIdx)
        {
            var hdr = this.HeaderOffset;
            var data = this.Data;
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            Debug.Assert(this.Shared != null);
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            Debug.Assert(nByte >= 0);  // Minimum cell size is 4
            Debug.Assert(this.FreeBytes >= nByte);
            Debug.Assert(this.NOverflows == 0);
            var usableSize = this.Shared.UsableSize; // Usable size of the page
            Debug.Assert(nByte < usableSize - 8);
            var nFrag = data[hdr + 7]; // Number of fragmented bytes on pPage
            Debug.Assert(this.CellOffset == hdr + 12 - 4 * this.Leaf);
            var gap = this.CellOffset + 2 * this.Cells; // First byte of gap between cell pointers and cell content
            var top = ConvertEx.Get2nz(data, hdr + 5); // First byte of cell content area
            if (gap > top)
                return SysEx.SQLITE_CORRUPT_BKPT();
            if (nFrag >= 60)
            {
                // Always defragment highly fragmented pages 
                var rc = defragmentPage();
                if (rc != RC.OK)
                    return rc;
                top = ConvertEx.Get2nz(data, hdr + 5);
            }
            else if (gap + 2 <= top)
            {
                // Search the freelist looking for a free slot big enough to satisfy the request. The allocation is made from the first free slot in
                // the list that is large enough to accomadate it.
                int pc;
                for (var addr = hdr + 1; (pc = ConvertEx.Get2(data, addr)) > 0; addr = pc)
                {
                    if (pc > usableSize - 4 || pc < addr + 4)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    var size = ConvertEx.Get2(data, pc + 2); // Size of free slot
                    if (size >= nByte)
                    {
                        var x = size - nByte;
                        if (x < 4)
                        {
                            // Remove the slot from the free-list. Update the number of fragmented bytes within the page.
                            data[addr + 0] = data[pc + 0];
                            data[addr + 1] = data[pc + 1];
                            data[hdr + 7] = (byte)(nFrag + x);
                        }
                        else if (size + pc > usableSize)
                            return SysEx.SQLITE_CORRUPT_BKPT();
                        else
                            // The slot remains on the free-list. Reduce its size to account for the portion used by the new allocation.
                            ConvertEx.Put2(data, pc + 2, x);
                        pIdx = pc + x;
                        return RC.OK;
                    }
                }
            }
            // Check to make sure there is enough space in the gap to satisfy the allocation.  If not, defragment.
            if (gap + 2 + nByte > top)
            {
                var rc = defragmentPage();
                if (rc != RC.OK)
                    return rc;
                top = ConvertEx.Get2nz(data, hdr + 5);
                Debug.Assert(gap + nByte <= top);
            }
            // Allocate memory from the gap in between the cell pointer array and the cell content area.  The btreeInitPage() call has already
            // validated the freelist.  Given that the freelist is valid, there is no way that the allocation can extend off the end of the page.
            // The Debug.Assert() below verifies the previous sentence.
            top -= nByte;
            ConvertEx.Put2(data, hdr + 5, top);
            Debug.Assert(top + nByte <= (int)this.Shared.UsableSize);
            pIdx = top;
            return RC.OK;
        }

        internal RC freeSpace(uint start, int size) { return freeSpace((int)start, size); }
        internal RC freeSpace(int start, int size)
        {
            var data = this.Data;
            Debug.Assert(this.Shared != null);
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            Debug.Assert(start >= this.HeaderOffset + 6 + this.ChildPtrSize);
            Debug.Assert((start + size) <= (int)this.Shared.UsableSize);
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            Debug.Assert(size >= 0);   // Minimum cell size is 4
            if (this.Shared.SecureDelete)
                // Overwrite deleted information with zeros when the secure_delete option is enabled
                Array.Clear(data, start, size);
            // Add the space back into the linked list of freeblocks.  Note that even though the freeblock list was checked by btreeInitPage(),
            // btreeInitPage() did not detect overlapping cells or freeblocks that overlapped cells.   Nor does it detect when the
            // cell content area exceeds the value in the page header.  If these situations arise, then subsequent insert operations might corrupt
            // the freelist.  So we do need to check for corruption while scanning the freelist.
            var hdr = this.HeaderOffset;
            var addr = hdr + 1;
            var iLast = (int)this.Shared.UsableSize - 4; // Largest possible freeblock offset
            Debug.Assert(start <= iLast);
            int pbegin;
            while ((pbegin = ConvertEx.Get2(data, addr)) < start && pbegin > 0)
            {
                if (pbegin < addr + 4)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                addr = pbegin;
            }
            if (pbegin > iLast)
                return SysEx.SQLITE_CORRUPT_BKPT();
            Debug.Assert(pbegin > addr || pbegin == 0);
            ConvertEx.Put2(data, addr, start);
            ConvertEx.Put2(data, start, pbegin);
            ConvertEx.Put2(data, start + 2, size);
            this.FreeBytes = (ushort)(this.FreeBytes + size);
            // Coalesce adjacent free blocks
            addr = hdr + 1;
            while ((pbegin = ConvertEx.Get2(data, addr)) > 0)
            {
                Debug.Assert(pbegin > addr);
                Debug.Assert(pbegin <= (int)this.Shared.UsableSize - 4);
                var pnext = ConvertEx.Get2(data, pbegin);
                var psize = ConvertEx.Get2(data, pbegin + 2);
                if (pbegin + psize + 3 >= pnext && pnext > 0)
                {
                    var frag = pnext - (pbegin + psize);
                    if ((frag < 0) || (frag > (int)data[hdr + 7]))
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    data[hdr + 7] -= (byte)frag;
                    var x = ConvertEx.Get2(data, pnext);
                    ConvertEx.Put2(data, pbegin, x);
                    x = pnext + ConvertEx.Get2(data, pnext + 2) - pbegin;
                    ConvertEx.Put2(data, pbegin + 2, x);
                }
                else
                    addr = pbegin;
            }
            // If the cell content area begins with a freeblock, remove it.
            if (data[hdr + 1] == data[hdr + 5] && data[hdr + 2] == data[hdr + 6])
            {
                pbegin = ConvertEx.Get2(data, hdr + 1);
                ConvertEx.Put2(data, hdr + 1, ConvertEx.Get2(data, pbegin));
                var top = ConvertEx.Get2(data, hdr + 5) + ConvertEx.Get2(data, pbegin + 2);
                ConvertEx.Put2(data, hdr + 5, top);
            }
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            return RC.OK;
        }

        internal RC decodeFlags(int flagByte)
        {
            Debug.Assert(this.HeaderOffset == (this.ID == 1 ? 100 : 0));
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            this.Leaf = (byte)(flagByte >> 3);
            Debug.Assert(Btree.PTF_LEAF == 1 << 3);
            flagByte &= ~Btree.PTF_LEAF;
            this.ChildPtrSize = (byte)(4 - 4 * this.Leaf);
            var pBt = this.Shared;
            if (flagByte == (Btree.PTF_LEAFDATA | Btree.PTF_INTKEY))
            {
                this.HasIntKey = true;
                this.HasData = this.Leaf;
                this.MaxLocal = pBt.MaxLeaf;
                this.MinLocal = pBt.MinLeaf;
            }
            else if (flagByte == Btree.PTF_ZERODATA)
            {
                this.HasIntKey = false;
                this.HasData = 0;
                this.MaxLocal = pBt.MaxLocal;
                this.MinLocal = pBt.MinLocal;
            }
            else
                return SysEx.SQLITE_CORRUPT_BKPT();
            return RC.OK;
        }

        internal RC btreeInitPage()
        {
            Debug.Assert(this.Shared != null);
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            Debug.Assert(this.ID == Pager.sqlite3PagerPagenumber(this.DbPage));
            Debug.Assert(this == Pager.sqlite3PagerGetExtra(this.DbPage));
            Debug.Assert(this.Data == Pager.sqlite3PagerGetData(this.DbPage));
            if (!this.HasInit)
            {
                var pBt = this.Shared; // The main btree structure
                var hdr = this.HeaderOffset; // Offset to beginning of page header
                var data = this.Data;
                if (decodeFlags(data[hdr]) != 0)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                Debug.Assert(pBt.PageSize >= 512 && pBt.PageSize <= 65536);
                this.MaskPage = (ushort)(pBt.PageSize - 1);
                this.NOverflows = 0;
                var usableSize = (int)pBt.UsableSize; // Amount of usable space on each page
                ushort cellOffset;    // Offset from start of page to first cell pointer
                this.CellOffset = (cellOffset = (ushort)(hdr + 12 - 4 * this.Leaf));
                var top = ConvertEx.Get2nz(data, hdr + 5); // First byte of the cell content area
                this.Cells = (ushort)(ConvertEx.Get2(data, hdr + 3));
                if (this.Cells > Btree.MX_CELL(pBt))
                    // To many cells for a single page.  The page must be corrupt
                    return SysEx.SQLITE_CORRUPT_BKPT();
                // A malformed database page might cause us to read past the end of page when parsing a cell.
                // The following block of code checks early to see if a cell extends past the end of a page boundary and causes SQLITE_CORRUPT to be
                // returned if it does.
                var iCellFirst = cellOffset + 2 * this.Cells; // First allowable cell or freeblock offset
                var iCellLast = usableSize - 4; // Last possible cell or freeblock offset
#if SQLITE_ENABLE_OVERSIZE_CELL_CHECK
                if (pPage.leaf == 0)
                    iCellLast--;
                for (var i = 0; i < pPage.nCell; i++)
                {
                    pc = (ushort)ConvertEx.get2byte(data, cellOffset + i * 2);
                    if (pc < iCellFirst || pc > iCellLast)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    var sz = cellSizePtr(pPage, data, pc);
                    if (pc + sz > usableSize)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                }
                if (pPage.leaf == 0)
                    iCellLast++;
#endif
                // Compute the total free space on the page
                var pc = (ushort)ConvertEx.Get2(data, hdr + 1); // Address of a freeblock within pPage.aData[]
                var nFree = (ushort)(data[hdr + 7] + top); // Number of unused bytes on the page
                while (pc > 0)
                {
                    if (pc < iCellFirst || pc > iCellLast)
                        // Start of free block is off the page
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    var next = (ushort)ConvertEx.Get2(data, pc);
                    var size = (ushort)ConvertEx.Get2(data, pc + 2);
                    if ((next > 0 && next <= pc + size + 3) || pc + size > usableSize)
                        // Free blocks must be in ascending order. And the last byte of the free-block must lie on the database page.
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    nFree = (ushort)(nFree + size);
                    pc = next;
                }
                // At this point, nFree contains the sum of the offset to the start of the cell-content area plus the number of free bytes within
                // the cell-content area. If this is greater than the usable-size of the page, then the page must be corrupted. This check also
                // serves to verify that the offset to the start of the cell-content area, according to the page header, lies within the page.
                if (nFree > usableSize)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                this.FreeBytes = (ushort)(nFree - iCellFirst);
                this.HasInit = true;
            }
            return RC.OK;
        }

        internal void zeroPage(int flags)
        {
            var data = this.Data;
            var pBt = this.Shared;
            var hdr = this.HeaderOffset;
            Debug.Assert(Pager.sqlite3PagerPagenumber(this.DbPage) == this.ID);
            Debug.Assert(Pager.sqlite3PagerGetExtra(this.DbPage) == this);
            Debug.Assert(Pager.sqlite3PagerGetData(this.DbPage) == data);
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            Debug.Assert(MutexEx.Held(pBt.Mutex));
            if (pBt.SecureDelete)
                Array.Clear(data, hdr, (int)(pBt.UsableSize - hdr));
            data[hdr] = (byte)flags;
            var first = (ushort)(hdr + 8 + 4 * ((flags & Btree.PTF_LEAF) == 0 ? 1 : 0));
            Array.Clear(data, hdr + 1, 4);
            data[hdr + 7] = 0;
            ConvertEx.Put2(data, hdr + 5, pBt.UsableSize);
            this.FreeBytes = (ushort)(pBt.UsableSize - first);
            decodeFlags(flags);
            this.HeaderOffset = hdr;
            this.CellOffset = first;
            this.NOverflows = 0;
            Debug.Assert(pBt.PageSize >= 512 && pBt.PageSize <= 65536);
            this.MaskPage = (ushort)(pBt.PageSize - 1);
            this.Cells = 0;
            this.HasInit = true;
        }

        internal static MemPage btreePageFromDbPage(DbPage pDbPage, Pgno pgno, BtShared pBt)
        {
            var pPage = (MemPage)Pager.sqlite3PagerGetExtra(pDbPage);
            pPage.Data = Pager.sqlite3PagerGetData(pDbPage);
            pPage.DbPage = pDbPage;
            pPage.Shared = pBt;
            pPage.ID = pgno;
            pPage.HeaderOffset = (byte)(pPage.ID == 1 ? 100 : 0);
            return pPage;
        }

        internal void releasePage()
        {
            if (this != null)
            {
                Debug.Assert(this.Data != null);
                Debug.Assert(this.Shared != null);
                // TODO -- find out why corrupt9 & diskfull fail on this tests 
                //Debug.Assert( Pager.sqlite3PagerGetExtra( pPage.pDbPage ) == pPage );
                //Debug.Assert( Pager.sqlite3PagerGetData( pPage.pDbPage ) == pPage.aData );
                Debug.Assert(MutexEx.Held(this.Shared.Mutex));
                Pager.sqlite3PagerUnref(this.DbPage);
            }
        }

#if DEBUG
        internal static void assertParentIndex(MemPage pParent, int iIdx, Pgno iChild)
        {
            Debug.Assert(iIdx <= pParent.Cells);
            if (iIdx == pParent.Cells)
                Debug.Assert(ConvertEx.Get4(pParent.Data, pParent.HeaderOffset + 8) == iChild);
            else
                Debug.Assert(ConvertEx.Get4(pParent.Data, pParent.FindCell(iIdx)) == iChild);
        }
#else
        internal static void assertParentIndex(MemPage pParent, int iIdx, Pgno iChild) { }
#endif

        internal RC fillInCell(byte[] pCell, byte[] pKey, long nKey, byte[] pData, int nData, int nZero, ref int pnSize)
        {
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            // pPage is not necessarily writeable since pCell might be auxiliary buffer space that is separate from the pPage buffer area
            // TODO -- Determine if the following Assert is needed under c#
            //Debug.Assert( pCell < pPage.aData || pCell >= &pPage.aData[pBt.pageSize] || sqlite3PagerIswriteable(pPage.pDbPage) );
            // Fill in the header.
            var nHeader = 0;
            if (this.Leaf == 0)
                nHeader += 4;
            if (this.HasData != 0)
                nHeader += (int)ConvertEx.PutVarint9(pCell, (uint)nHeader, (int)(nData + nZero));
            else
                nData = nZero = 0;
            nHeader += ConvertEx.PutVarint9L(pCell, (uint)nHeader, (ulong)nKey);
            var info = new CellInfo();
            btreeParseCellPtr(pCell, ref info);
            Debug.Assert(info.nHeader == nHeader);
            Debug.Assert(info.nKey == nKey);
            Debug.Assert(info.nData == (uint)(nData + nZero));
            // Fill in the payload
            var nPayload = nData + nZero;
            byte[] pSrc;
            int nSrc;
            if (this.HasIntKey)
            {
                pSrc = pData;
                nSrc = nData;
                nData = 0;
            }
            else
            {
                if (Check.NEVER(nKey > 0x7fffffff || pKey == null))
                    return SysEx.SQLITE_CORRUPT_BKPT();
                nPayload += (int)nKey;
                pSrc = pKey;
                nSrc = (int)nKey;
            }
            pnSize = info.nSize;
            var spaceLeft = (int)info.nLocal;
            var pPayload = pCell;
            var pPayloadIndex = nHeader;
            var pPrior = pCell;
            var pPriorIndex = (int)info.iOverflow;
            var pBt = this.Shared;
            Pgno pgnoOvfl = 0;
            MemPage pToRelease = null;
            while (nPayload > 0)
            {
                if (spaceLeft == 0)
                {
#if !SQLITE_OMIT_AUTOVACUUM
                    var pgnoPtrmap = pgnoOvfl; // Overflow page pointer-map entry page
                    if (pBt.AutoVacuum)
                    {
                        do
                            pgnoOvfl++;
                        while (PTRMAP_ISPAGE(pBt, pgnoOvfl) || pgnoOvfl == PENDING_BYTE_PAGE(pBt));
                    }
#endif
                    MemPage pOvfl = null;
                    var rc = pBt.allocateBtreePage(ref pOvfl, ref pgnoOvfl, pgnoOvfl, 0);
#if !SQLITE_OMIT_AUTOVACUUM
                    // If the database supports auto-vacuum, and the second or subsequent overflow page is being allocated, add an entry to the pointer-map for that page now.
                    // If this is the first overflow page, then write a partial entry to the pointer-map. If we write nothing to this pointer-map slot,
                    // then the optimistic overflow chain processing in clearCell() may misinterpret the uninitialised values and delete the
                    // wrong pages from the database.
                    if (pBt.AutoVacuum && rc == RC.OK)
                    {
                        var eType = (pgnoPtrmap != 0 ? PTRMAP.OVERFLOW2 : PTRMAP.OVERFLOW1);
                        pBt.ptrmapPut(pgnoOvfl, eType, pgnoPtrmap, ref rc);
                        if (rc != RC.OK)
                            pOvfl.releasePage();
                    }
#endif
                    if (rc != RC.OK)
                    {
                        pToRelease.releasePage();
                        return rc;
                    }
                    // If pToRelease is not zero than pPrior points into the data area of pToRelease.  Make sure pToRelease is still writeable.
                    Debug.Assert(pToRelease == null || Pager.sqlite3PagerIswriteable(pToRelease.DbPage));
                    // If pPrior is part of the data area of pPage, then make sure pPage is still writeable
                    // TODO -- Determine if the following Assert is needed under c#
                    //Debug.Assert( pPrior < pPage.aData || pPrior >= &pPage.aData[pBt.pageSize] || sqlite3PagerIswriteable(pPage.pDbPage) );
                    ConvertEx.Put4L(pPrior, pPriorIndex, pgnoOvfl);
                    pToRelease.releasePage();
                    pToRelease = pOvfl;
                    pPrior = pOvfl.Data;
                    pPriorIndex = 0;
                    ConvertEx.Put4(pPrior, 0);
                    pPayload = pOvfl.Data;
                    pPayloadIndex = 4;
                    spaceLeft = (int)pBt.UsableSize - 4;
                }
                var n = nPayload;
                if (n > spaceLeft)
                    n = spaceLeft;
                // If pToRelease is not zero than pPayload points into the data area of pToRelease.  Make sure pToRelease is still writeable.
                Debug.Assert(pToRelease == null || Pager.sqlite3PagerIswriteable(pToRelease.DbPage));
                // If pPayload is part of the data area of pPage, then make sure pPage is still writeable
                // TODO -- Determine if the following Assert is needed under c#
                //Debug.Assert( pPayload < pPage.aData || pPayload >= &pPage.aData[pBt.pageSize] || sqlite3PagerIswriteable(pPage.pDbPage) );
                var pSrcIndex = 0;
                if (nSrc > 0)
                {
                    if (n > nSrc)
                        n = nSrc;
                    Debug.Assert(pSrc != null);
                    Buffer.BlockCopy(pSrc, pSrcIndex, pPayload, pPayloadIndex, n);
                }
                else
                {
                    var pZeroBlob = MallocEx.sqlite3Malloc(n);
                    Buffer.BlockCopy(pZeroBlob, 0, pPayload, pPayloadIndex, n);
                }
                nPayload -= n;
                pPayloadIndex += n;
                pSrcIndex += n;
                nSrc -= n;
                spaceLeft -= n;
                if (nSrc == 0)
                {
                    nSrc = nData;
                    pSrc = pData;
                }
            }
            pToRelease.releasePage();
            return RC.OK;
        }

        internal void dropCell(int idx, int sz, ref RC pRC)
        {
            if (pRC != RC.OK)
                return;
            Debug.Assert(idx >= 0 && idx < this.Cells);
#if DEBUG
            Debug.Assert(sz == cellSize(idx));
#endif
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            var data = this.Data;
            var ptr = this.CellOffset + 2 * idx; // Used to move bytes around within data[]
            var pc = (uint)ConvertEx.Get2(data, ptr); // Offset to cell content of cell being deleted
            var hdr = this.HeaderOffset; // Beginning of the header.  0 most pages.  100 page 1
            if (pc < (uint)ConvertEx.Get2(data, hdr + 5) || pc + sz > this.Shared.UsableSize)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                return;
            }
            var rc = freeSpace(pc, sz);
            if (rc != RC.OK)
            {
                pRC = rc;
                return;
            }
            Buffer.BlockCopy(data, ptr + 2, data, ptr, (this.Cells - 1 - idx) * 2);
            this.Cells--;
            data[this.HeaderOffset + 3] = (byte)(this.Cells >> 8);
            data[this.HeaderOffset + 4] = (byte)(this.Cells);
            this.FreeBytes += 2;
        }

        internal void insertCell(int i, byte[] pCell, int sz, byte[] pTemp, Pgno iChild, ref RC pRC)
        {
            var nSkip = (iChild != 0 ? 4 : 0);
            if (pRC != RC.OK)
                return;
            Debug.Assert(i >= 0 && i <= this.Cells + this.NOverflows);
            Debug.Assert(this.Cells <= Btree.MX_CELL(this.Shared) && Btree.MX_CELL(this.Shared) <= 10921);
            Debug.Assert(this.NOverflows <= this.Overflows.Length);
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            // The cell should normally be sized correctly.  However, when moving a malformed cell from a leaf page to an interior page, if the cell size
            // wanted to be less than 4 but got rounded up to 4 on the leaf, then size might be less than 8 (leaf-size + pointer) on the interior node.  Hence
            // the term after the || in the following assert().
            Debug.Assert(sz == cellSizePtr(pCell) || (sz == 8 && iChild > 0));
            if (this.NOverflows != 0 || sz + 2 > this.FreeBytes)
            {
                if (pTemp != null)
                {
                    Buffer.BlockCopy(pCell, nSkip, pTemp, nSkip, sz - nSkip);
                    pCell = pTemp;
                }
                if (iChild != 0)
                    ConvertEx.Put4L(pCell, iChild);
                var j = this.NOverflows++;
                Debug.Assert(j < this.Overflows.Length);
                this.Overflows[j].Cell = pCell;
                this.Overflows[j].Index = (ushort)i;
            }
            else
            {
                var rc = Pager.sqlite3PagerWrite(this.DbPage);
                if (rc != RC.OK)
                {
                    pRC = rc;
                    return;
                }
                Debug.Assert(Pager.sqlite3PagerIswriteable(this.DbPage));
                var data = this.Data; // The content of the whole page
                var cellOffset = (int)this.CellOffset; // Address of first cell pointer in data[]
                var end = cellOffset + 2 * this.Cells; // First byte past the last cell pointer in data[]
                var ins = cellOffset + 2 * i; // Index in data[] where new cell pointer is inserted
                int idx = 0; // Where to write new cell content in data[]
                rc = allocateSpace(sz, ref idx);
                if (rc != RC.OK)
                {
                    pRC = rc;
                    return;
                }
                // The allocateSpace() routine guarantees the following two properties if it returns success
                Debug.Assert(idx >= end + 2);
                Debug.Assert(idx + sz <= (int)this.Shared.UsableSize);
                this.Cells++;
                this.FreeBytes -= (ushort)(2 + sz);
                Buffer.BlockCopy(pCell, nSkip, data, idx + nSkip, sz - nSkip);
                if (iChild != 0)
                    ConvertEx.Put4L(data, idx, iChild);
                for (var j = end; j > ins; j -= 2)
                {
                    data[j + 0] = data[j - 2];
                    data[j + 1] = data[j - 1];
                }
                ConvertEx.Put2(data, ins, idx);
                ConvertEx.Put2(data, this.HeaderOffset + 3, this.Cells);
#if !SQLITE_OMIT_AUTOVACUUM
                if (this.Shared.AutoVacuum)
                    // The cell may contain a pointer to an overflow page. If so, write the entry for the overflow page into the pointer map.
                    ptrmapPutOvflPtr(pCell, ref pRC);
#endif
            }
        }

        internal void freePage(ref RC pRC)
        {
            if (pRC == RC.OK)
                pRC = this.Shared.freePage2(this, this.ID);
        }

        internal RC clearCell(int pCell)
        {
            Debug.Assert(MutexEx.Held(this.Shared.Mutex));
            var info = new CellInfo();
            btreeParseCellPtr(pCell, ref info);
            if (info.iOverflow == 0)
                return RC.OK;  // No overflow pages. Return without doing anything
            var pBt = this.Shared;
            var ovflPgno = (Pgno)ConvertEx.Get4(this.Data, pCell, info.iOverflow);
            Debug.Assert(pBt.UsableSize > 4);
            var ovflPageSize = (uint)(pBt.UsableSize - 4);
            var nOvfl = (int)((info.nPayload - info.nLocal + ovflPageSize - 1) / ovflPageSize);
            Debug.Assert(ovflPgno == 0 || nOvfl > 0);
            RC rc;
            while (nOvfl-- != 0)
            {
                Pgno iNext = 0;
                MemPage pOvfl = null;
                if (ovflPgno < 2 || ovflPgno > pBt.btreePagecount())
                    // 0 is not a legal page number and page 1 cannot be an overflow page. Therefore if ovflPgno<2 or past the end of the file the database must be corrupt.
                    return SysEx.SQLITE_CORRUPT_BKPT();
                if (nOvfl != 0)
                {
                    rc = pBt.getOverflowPage(ovflPgno, out pOvfl, out iNext);
                    if (rc != RC.OK)
                        return rc;
                }
                if ((pOvfl != null || ((pOvfl = pBt.btreePageLookup(ovflPgno)) != null)) && Pager.sqlite3PagerPageRefcount(pOvfl.DbPage) != 1)
                    // There is no reason any cursor should have an outstanding reference to an overflow page belonging to a cell that is being deleted/updated.
                    // So if there exists more than one reference to this page, then it must not really be an overflow page and the database must be corrupt. 
                    // It is helpful to detect this before calling freePage2(), as freePage2() may zero the page contents if secure-delete mode is
                    // enabled. If this 'overflow' page happens to be a page that the caller is iterating through or using in some other way, this
                    // can be problematic.
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                else
                    rc = pBt.freePage2(pOvfl, ovflPgno);
                if (pOvfl != null)
                    Pager.sqlite3PagerUnref(pOvfl.DbPage);
                if (rc != RC.OK)
                    return rc;
                ovflPgno = iNext;
            }
            return RC.OK;
        }
    }
}
