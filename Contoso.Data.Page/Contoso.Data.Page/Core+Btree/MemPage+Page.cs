using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class MemPage
    {
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

        internal void freePage(ref RC pRC)
        {
            if (pRC == RC.OK)
                pRC = this.Shared.freePage2(this, this.ID);
        }
    }
}
