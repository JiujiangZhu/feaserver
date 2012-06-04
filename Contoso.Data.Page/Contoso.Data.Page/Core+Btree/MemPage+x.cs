﻿using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;

namespace Contoso.Core
{
    public partial class Btree
    {
        internal static int findCell(MemPage pPage, int iCell) { return ConvertEx.get2byte(pPage.aData, pPage.cellOffset + 2 * (iCell)); }
        internal static byte[] findCellv2(byte[] pPage, ushort iCell, ushort O, int I) { Debugger.Break(); return pPage; }
        internal static int findOverflowCell(MemPage pPage, int iCell)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            for (var i = pPage.nOverflow - 1; i >= 0; i--)
            {
                var pOvfl = pPage.aOvfl[i];
                var k = pOvfl.idx;
                if (k <= iCell)
                {
                    if (k == iCell)
                        return -i - 1; // Negative Offset means overflow cells
                    iCell--;
                }
            }
            return findCell(pPage, iCell);
        }

        internal static void btreeParseCellPtr(MemPage pPage, int iCell, ref CellInfo pInfo) { btreeParseCellPtr(pPage, pPage.aData, iCell, ref pInfo); }
        internal static void btreeParseCellPtr(MemPage pPage, byte[] pCell, ref CellInfo pInfo) { btreeParseCellPtr(pPage, pCell, 0, ref pInfo); }
        internal static void btreeParseCellPtr(MemPage pPage, byte[] pCell, int iCell, ref CellInfo pInfo)
        {
            var nPayload = (uint)0;    // Number of bytes of cell payload
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            if (pInfo.pCell != pCell)
                pInfo.pCell = pCell;
            pInfo.iCell = iCell;
            Debug.Assert(pPage.leaf == 0 || pPage.leaf == 1);
            var n = (ushort)pPage.childPtrSize; // Number bytes in cell content header
            Debug.Assert(n == 4 - 4 * pPage.leaf);
            if (pPage.intKey != 0)
            {
                if (pPage.hasData != 0)
                    n += (ushort)ConvertEx.getVarint32(pCell, iCell + n, out nPayload);
                else
                    nPayload = 0;
                n += (ushort)ConvertEx.getVarint(pCell, iCell + n, out pInfo.nKey);
                pInfo.nData = nPayload;
            }
            else
            {
                pInfo.nData = 0;
                n += (ushort)ConvertEx.getVarint32(pCell, iCell + n, out nPayload);
                pInfo.nKey = nPayload;
            }
            pInfo.nPayload = nPayload;
            pInfo.nHeader = n;
            if (likely(nPayload <= pPage.maxLocal))
            {
                // This is the (easy) common case where the entire payload fits on the local page.  No overflow is required.
                if ((pInfo.nSize = (ushort)(n + nPayload)) < 4)
                    pInfo.nSize = 4;
                pInfo.nLocal = (ushort)nPayload;
                pInfo.iOverflow = 0;
            }
            else
            {
                // If the payload will not fit completely on the local page, we have to decide how much to store locally and how much to spill onto
                // overflow pages.  The strategy is to minimize the amount of unused space on overflow pages while keeping the amount of local storage
                // in between minLocal and maxLocal.
                // Warning:  changing the way overflow payload is distributed in any way will result in an incompatible file format.
                var minLocal = (int)pPage.minLocal; // Minimum amount of payload held locally
                var maxLocal = (int)pPage.maxLocal;// Maximum amount of payload held locally
                var surplus = (int)(minLocal + (nPayload - minLocal) % (pPage.pBt.usableSize - 4));// Overflow payload available for local storage
                pInfo.nLocal = (surplus <= maxLocal ? (ushort)surplus : (ushort)minLocal);
                pInfo.iOverflow = (ushort)(pInfo.nLocal + n);
                pInfo.nSize = (ushort)(pInfo.iOverflow + 4);
            }
        }

        internal static void parseCell(MemPage pPage, int iCell, ref CellInfo pInfo) { btreeParseCellPtr(pPage, findCell(pPage, iCell), ref pInfo); }
        internal static void btreeParseCell(MemPage pPage, int iCell, ref CellInfo pInfo) { parseCell(pPage, iCell, ref pInfo); }

        internal static ushort cellSizePtr(MemPage pPage, int iCell)
        {
            var info = new CellInfo();
            var pCell = new byte[13];
            // Minimum Size = (2 bytes of Header  or (4) Child Pointer) + (maximum of) 9 bytes data
            if (iCell < 0)// Overflow Cell
                Buffer.BlockCopy(pPage.aOvfl[-(iCell + 1)].pCell, 0, pCell, 0, pCell.Length < pPage.aOvfl[-(iCell + 1)].pCell.Length ? pCell.Length : pPage.aOvfl[-(iCell + 1)].pCell.Length);
            else if (iCell >= pPage.aData.Length + 1 - pCell.Length)
                Buffer.BlockCopy(pPage.aData, iCell, pCell, 0, pPage.aData.Length - iCell);
            else
                Buffer.BlockCopy(pPage.aData, iCell, pCell, 0, pCell.Length);
            btreeParseCellPtr(pPage, pCell, ref info);
            return info.nSize;
        }
        internal static ushort cellSizePtr(MemPage pPage, byte[] pCell, int offset)
        {
            var info = new CellInfo();
            info.pCell = MallocEx.sqlite3Malloc(pCell.Length);
            Buffer.BlockCopy(pCell, offset, info.pCell, 0, pCell.Length - offset);
            btreeParseCellPtr(pPage, info.pCell, ref info);
            return info.nSize;
        }
        internal static ushort cellSizePtr(MemPage pPage, byte[] pCell)
        {
            int _pIter = pPage.childPtrSize;
            uint nSize = 0;

#if DEBUG
            // The value returned by this function should always be the same as the (CellInfo.nSize) value found by doing a full parse of the
            // cell. If SQLITE_DEBUG is defined, an Debug.Assert() at the bottom of this function verifies that this invariant is not violated. */
            var debuginfo = new CellInfo();
            btreeParseCellPtr(pPage, pCell, ref debuginfo);
#else
            var debuginfo = new CellInfo();
#endif
            if (pPage.intKey != 0)
            {
                if (pPage.hasData != 0)
                    _pIter += ConvertEx.getVarint32(pCell, out nSize);
                else
                    nSize = 0;
                // pIter now points at the 64-bit integer key value, a variable length integer. The following block moves pIter to point at the first byte
                // past the end of the key value. */
                var pEnd = _pIter + 9;
                while (((pCell[_pIter++]) & 0x80) != 0 && _pIter < pEnd) ;
            }
            else
                _pIter += ConvertEx.getVarint32(pCell, _pIter, out nSize);
            if (nSize > pPage.maxLocal)
            {
                var minLocal = (int)pPage.minLocal;
                nSize = (ushort)(minLocal + (nSize - minLocal) % (pPage.pBt.usableSize - 4));
                if (nSize > pPage.maxLocal)
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
        internal static ushort cellSize(MemPage pPage, int iCell) { return cellSizePtr(pPage, findCell(pPage, iCell)); }
#else
        internal static int cellSize(MemPage pPage, int iCell) { return -1; }
#endif

#if !SQLITE_OMIT_AUTOVACUUM
        internal static void ptrmapPutOvflPtr(MemPage pPage, int pCell, ref int pRC)
        {
            if (pRC != 0)
                return;
            var info = new CellInfo();
            Debug.Assert(pCell != 0);
            btreeParseCellPtr(pPage, pCell, ref info);
            Debug.Assert((info.nData + (pPage.intKey != 0 ? 0 : info.nKey)) == info.nPayload);
            if (info.iOverflow != 0)
            {
                Pgno ovfl = ConvertEx.sqlite3Get4byte(pPage.aData, pCell, info.iOverflow);
                ptrmapPut(pPage.pBt, ovfl, PTRMAP_OVERFLOW1, pPage.pgno, ref pRC);
            }
        }

        internal static void ptrmapPutOvflPtr(MemPage pPage, byte[] pCell, ref int pRC)
        {
            if (pRC != 0)
                return;
            var info = new CellInfo();
            Debug.Assert(pCell != null);
            btreeParseCellPtr(pPage, pCell, ref info);
            Debug.Assert((info.nData + (pPage.intKey != 0 ? 0 : info.nKey)) == info.nPayload);
            if (info.iOverflow != 0)
            {
                Pgno ovfl = ConvertEx.sqlite3Get4byte(pCell, info.iOverflow);
                ptrmapPut(pPage.pBt, ovfl, PTRMAP_OVERFLOW1, pPage.pgno, ref pRC);
            }
        }
#endif

        internal static SQLITE defragmentPage(MemPage pPage)
        {
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            Debug.Assert(pPage.pBt != null);
            Debug.Assert(pPage.pBt.usableSize <= Pager.SQLITE_MAX_PAGE_SIZE);
            Debug.Assert(pPage.nOverflow == 0);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            var temp = pPage.pBt.pPager.sqlite3PagerTempSpace();// Temp area for cell content
            var data = pPage.aData; // The page data
            var hdr = pPage.hdrOffset; // Offset to the page header
            var cellOffset = pPage.cellOffset;// Offset to the cell pointer array
            var nCell = pPage.nCell;// Number of cells on the page
            Debug.Assert(nCell == ConvertEx.get2byte(data, hdr + 3));
            var usableSize = (int)pPage.pBt.usableSize;// Number of usable bytes on a page
            var cbrk = (int)ConvertEx.get2byte(data, hdr + 5);// Offset to the cell content area
            Buffer.BlockCopy(data, cbrk, temp, cbrk, usableSize - cbrk);
            cbrk = usableSize;
            var iCellFirst = cellOffset + 2 * nCell; // First allowable cell index
            var iCellLast = usableSize - 4; // Last possible cell index
            for (var i = 0; i < nCell; i++)
            {
                var pAddr = cellOffset + i * 2; // The i-th cell pointer
                var pc = ConvertEx.get2byte(data, pAddr); // Address of a i-th cell 
#if !SQLITE_ENABLE_OVERSIZE_CELL_CHECK
                // These conditions have already been verified in btreeInitPage() if SQLITE_ENABLE_OVERSIZE_CELL_CHECK is defined
                if (pc < iCellFirst || pc > iCellLast)
                    return SysEx.SQLITE_CORRUPT_BKPT();
#endif
                Debug.Assert(pc >= iCellFirst && pc <= iCellLast);
                var size = cellSizePtr(pPage, temp, pc); // Size of a cell 
                cbrk -= size;
                if (cbrk < iCellFirst || pc + size > usableSize)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                Debug.Assert(cbrk + size <= usableSize && cbrk >= iCellFirst);
                Buffer.BlockCopy(temp, pc, data, cbrk, size);
                ConvertEx.put2byte(data, pAddr, cbrk);
            }
            Debug.Assert(cbrk >= iCellFirst);
            ConvertEx.put2byte(data, hdr + 5, cbrk);
            data[hdr + 1] = 0;
            data[hdr + 2] = 0;
            data[hdr + 7] = 0;
            var addr = cellOffset + 2 * nCell; // Offset of first byte after cell pointer array 
            Array.Clear(data, addr, cbrk - addr);
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            return (cbrk - iCellFirst != pPage.nFree ? SysEx.SQLITE_CORRUPT_BKPT() : SQLITE.OK);
        }

        internal static SQLITE allocateSpace(MemPage pPage, int nByte, ref int pIdx)
        {
            var hdr = pPage.hdrOffset;
            var data = pPage.aData;
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            Debug.Assert(pPage.pBt != null);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(nByte >= 0);  // Minimum cell size is 4
            Debug.Assert(pPage.nFree >= nByte);
            Debug.Assert(pPage.nOverflow == 0);
            var usableSize = pPage.pBt.usableSize; // Usable size of the page
            Debug.Assert(nByte < usableSize - 8);
            var nFrag = data[hdr + 7]; // Number of fragmented bytes on pPage
            Debug.Assert(pPage.cellOffset == hdr + 12 - 4 * pPage.leaf);
            var gap = pPage.cellOffset + 2 * pPage.nCell; // First byte of gap between cell pointers and cell content
            var top = ConvertEx.get2byteNotZero(data, hdr + 5); // First byte of cell content area
            if (gap > top)
                return SysEx.SQLITE_CORRUPT_BKPT();
            if (nFrag >= 60)
            {
                // Always defragment highly fragmented pages 
                var rc = defragmentPage(pPage);
                if (rc != SQLITE.OK)
                    return rc;
                top = ConvertEx.get2byteNotZero(data, hdr + 5);
            }
            else if (gap + 2 <= top)
            {
                // Search the freelist looking for a free slot big enough to satisfy the request. The allocation is made from the first free slot in
                // the list that is large enough to accomadate it.
                int pc;
                for (var addr = hdr + 1; (pc = ConvertEx.get2byte(data, addr)) > 0; addr = pc)
                {
                    if (pc > usableSize - 4 || pc < addr + 4)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    var size = ConvertEx.get2byte(data, pc + 2); // Size of free slot
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
                            ConvertEx.put2byte(data, pc + 2, x);
                        pIdx = pc + x;
                        return SQLITE.OK;
                    }
                }
            }
            // Check to make sure there is enough space in the gap to satisfy the allocation.  If not, defragment.
            if (gap + 2 + nByte > top)
            {
                var rc = defragmentPage(pPage);
                if (rc != SQLITE.OK)
                    return rc;
                top = ConvertEx.get2byteNotZero(data, hdr + 5);
                Debug.Assert(gap + nByte <= top);
            }
            // Allocate memory from the gap in between the cell pointer array and the cell content area.  The btreeInitPage() call has already
            // validated the freelist.  Given that the freelist is valid, there is no way that the allocation can extend off the end of the page.
            // The Debug.Assert() below verifies the previous sentence.
            top -= nByte;
            ConvertEx.put2byte(data, hdr + 5, top);
            Debug.Assert(top + nByte <= (int)pPage.pBt.usableSize);
            pIdx = top;
            return SQLITE.OK;
        }

        internal static SQLITE freeSpace(MemPage pPage, uint start, int size) { return freeSpace(pPage, (int)start, size); }
        internal static SQLITE freeSpace(MemPage pPage, int start, int size)
        {
            var data = pPage.aData;
            Debug.Assert(pPage.pBt != null);
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            Debug.Assert(start >= pPage.hdrOffset + 6 + pPage.childPtrSize);
            Debug.Assert((start + size) <= (int)pPage.pBt.usableSize);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(size >= 0);   // Minimum cell size is 4
            if (pPage.pBt.secureDelete)
                // Overwrite deleted information with zeros when the secure_delete option is enabled
                Array.Clear(data, start, size);
            // Add the space back into the linked list of freeblocks.  Note that even though the freeblock list was checked by btreeInitPage(),
            // btreeInitPage() did not detect overlapping cells or freeblocks that overlapped cells.   Nor does it detect when the
            // cell content area exceeds the value in the page header.  If these situations arise, then subsequent insert operations might corrupt
            // the freelist.  So we do need to check for corruption while scanning the freelist.
            var hdr = pPage.hdrOffset;
            var addr = hdr + 1;
            var iLast = (int)pPage.pBt.usableSize - 4; // Largest possible freeblock offset
            Debug.Assert(start <= iLast);
            int pbegin;
            while ((pbegin = ConvertEx.get2byte(data, addr)) < start && pbegin > 0)
            {
                if (pbegin < addr + 4)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                addr = pbegin;
            }
            if (pbegin > iLast)
                return SysEx.SQLITE_CORRUPT_BKPT();
            Debug.Assert(pbegin > addr || pbegin == 0);
            ConvertEx.put2byte(data, addr, start);
            ConvertEx.put2byte(data, start, pbegin);
            ConvertEx.put2byte(data, start + 2, size);
            pPage.nFree = (ushort)(pPage.nFree + size);
            // Coalesce adjacent free blocks
            addr = hdr + 1;
            while ((pbegin = ConvertEx.get2byte(data, addr)) > 0)
            {
                Debug.Assert(pbegin > addr);
                Debug.Assert(pbegin <= (int)pPage.pBt.usableSize - 4);
                var pnext = ConvertEx.get2byte(data, pbegin);
                var psize = ConvertEx.get2byte(data, pbegin + 2);
                if (pbegin + psize + 3 >= pnext && pnext > 0)
                {
                    var frag = pnext - (pbegin + psize);
                    if ((frag < 0) || (frag > (int)data[hdr + 7]))
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    data[hdr + 7] -= (byte)frag;
                    var x = ConvertEx.get2byte(data, pnext);
                    ConvertEx.put2byte(data, pbegin, x);
                    x = pnext + ConvertEx.get2byte(data, pnext + 2) - pbegin;
                    ConvertEx.put2byte(data, pbegin + 2, x);
                }
                else
                    addr = pbegin;
            }
            // If the cell content area begins with a freeblock, remove it.
            if (data[hdr + 1] == data[hdr + 5] && data[hdr + 2] == data[hdr + 6])
            {
                pbegin = ConvertEx.get2byte(data, hdr + 1);
                ConvertEx.put2byte(data, hdr + 1, ConvertEx.get2byte(data, pbegin));
                var top = ConvertEx.get2byte(data, hdr + 5) + ConvertEx.get2byte(data, pbegin + 2);
                ConvertEx.put2byte(data, hdr + 5, top);
            }
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            return SQLITE.OK;
        }

        internal static SQLITE decodeFlags(MemPage pPage, int flagByte)
        {
            Debug.Assert(pPage.hdrOffset == (pPage.pgno == 1 ? 100 : 0));
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            pPage.leaf = (byte)(flagByte >> 3);
            Debug.Assert(PTF_LEAF == 1 << 3);
            flagByte &= ~PTF_LEAF;
            pPage.childPtrSize = (byte)(4 - 4 * pPage.leaf);
            var pBt = pPage.pBt;
            if (flagByte == (PTF_LEAFDATA | PTF_INTKEY))
            {
                pPage.intKey = 1;
                pPage.hasData = pPage.leaf;
                pPage.maxLocal = pBt.maxLeaf;
                pPage.minLocal = pBt.minLeaf;
            }
            else if (flagByte == PTF_ZERODATA)
            {
                pPage.intKey = 0;
                pPage.hasData = 0;
                pPage.maxLocal = pBt.maxLocal;
                pPage.minLocal = pBt.minLocal;
            }
            else
                return SysEx.SQLITE_CORRUPT_BKPT();
            return SQLITE.OK;
        }

        internal static SQLITE btreeInitPage(MemPage pPage)
        {
            Debug.Assert(pPage.pBt != null);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(pPage.pgno == Pager.sqlite3PagerPagenumber(pPage.pDbPage));
            Debug.Assert(pPage == Pager.sqlite3PagerGetExtra(pPage.pDbPage));
            Debug.Assert(pPage.aData == Pager.sqlite3PagerGetData(pPage.pDbPage));
            if (pPage.isInit == 0)
            {
                var pBt = pPage.pBt; // The main btree structure
                var hdr = pPage.hdrOffset; // Offset to beginning of page header
                var data = pPage.aData;
                if (decodeFlags(pPage, data[hdr]) != 0)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                Debug.Assert(pBt.pageSize >= 512 && pBt.pageSize <= 65536);
                pPage.maskPage = (ushort)(pBt.pageSize - 1);
                pPage.nOverflow = 0;
                var usableSize = (int)pBt.usableSize; // Amount of usable space on each page
                ushort cellOffset;    // Offset from start of page to first cell pointer
                pPage.cellOffset = (cellOffset = (ushort)(hdr + 12 - 4 * pPage.leaf));
                var top = ConvertEx.get2byteNotZero(data, hdr + 5); // First byte of the cell content area
                pPage.nCell = (ushort)(ConvertEx.get2byte(data, hdr + 3));
                if (pPage.nCell > MX_CELL(pBt))
                    // To many cells for a single page.  The page must be corrupt
                    return SysEx.SQLITE_CORRUPT_BKPT();
                // A malformed database page might cause us to read past the end of page when parsing a cell.
                // The following block of code checks early to see if a cell extends past the end of a page boundary and causes SQLITE_CORRUPT to be
                // returned if it does.
                var iCellFirst = cellOffset + 2 * pPage.nCell; // First allowable cell or freeblock offset
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
                var pc = (ushort)ConvertEx.get2byte(data, hdr + 1); // Address of a freeblock within pPage.aData[]
                var nFree = (ushort)(data[hdr + 7] + top); // Number of unused bytes on the page
                while (pc > 0)
                {
                    if (pc < iCellFirst || pc > iCellLast)
                        // Start of free block is off the page
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    var next = (ushort)ConvertEx.get2byte(data, pc);
                    var size = (ushort)ConvertEx.get2byte(data, pc + 2);
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
                pPage.nFree = (ushort)(nFree - iCellFirst);
                pPage.isInit = 1;
            }
            return SQLITE.OK;
        }

        internal static void zeroPage(MemPage pPage, int flags)
        {
            var data = pPage.aData;
            var pBt = pPage.pBt;
            var hdr = pPage.hdrOffset;
            Debug.Assert(Pager.sqlite3PagerPagenumber(pPage.pDbPage) == pPage.pgno);
            Debug.Assert(Pager.sqlite3PagerGetExtra(pPage.pDbPage) == pPage);
            Debug.Assert(Pager.sqlite3PagerGetData(pPage.pDbPage) == data);
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            if (pBt.secureDelete)
                Array.Clear(data, hdr, (int)(pBt.usableSize - hdr));
            data[hdr] = (byte)flags;
            var first = (ushort)(hdr + 8 + 4 * ((flags & PTF_LEAF) == 0 ? 1 : 0));
            Array.Clear(data, hdr + 1, 4);
            data[hdr + 7] = 0;
            ConvertEx.put2byte(data, hdr + 5, pBt.usableSize);
            pPage.nFree = (ushort)(pBt.usableSize - first);
            decodeFlags(pPage, flags);
            pPage.hdrOffset = hdr;
            pPage.cellOffset = first;
            pPage.nOverflow = 0;
            Debug.Assert(pBt.pageSize >= 512 && pBt.pageSize <= 65536);
            pPage.maskPage = (ushort)(pBt.pageSize - 1);
            pPage.nCell = 0;
            pPage.isInit = 1;
        }

        internal static MemPage btreePageFromDbPage(DbPage pDbPage, Pgno pgno, BtShared pBt)
        {
            var pPage = (MemPage)Pager.sqlite3PagerGetExtra(pDbPage);
            pPage.aData = Pager.sqlite3PagerGetData(pDbPage);
            pPage.pDbPage = pDbPage;
            pPage.pBt = pBt;
            pPage.pgno = pgno;
            pPage.hdrOffset = (byte)(pPage.pgno == 1 ? 100 : 0);
            return pPage;
        }

        internal static SQLITE btreeGetPage(BtShared pBt, Pgno pgno, ref MemPage ppPage, int noContent)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            DbPage pDbPage = null;
            var rc = pBt.pPager.sqlite3PagerAcquire(pgno, ref pDbPage, (byte)noContent);
            if (rc != SQLITE.OK)
                return rc;
            ppPage = btreePageFromDbPage(pDbPage, pgno, pBt);
            return SQLITE.OK;
        }

        internal static MemPage btreePageLookup(BtShared pBt, Pgno pgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            var pDbPage = pBt.pPager.sqlite3PagerLookup(pgno);
            return (pDbPage ? btreePageFromDbPage(pDbPage, pgno, pBt) : null);
        }

        internal static Pgno btreePagecount(BtShared pBt) { return pBt.nPage; }

        internal static Pgno sqlite3BtreeLastPage(Btree p)
        {
            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            Debug.Assert(((p.pBt.nPage) & 0x8000000) == 0);
            return (Pgno)btreePagecount(p.pBt);
        }

        internal static SQLITE getAndInitPage(BtShared pBt, Pgno pgno, ref MemPage ppPage)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            SQLITE rc;
            if (pgno > btreePagecount(pBt))
                rc = SysEx.SQLITE_CORRUPT_BKPT();
            else
            {
                rc = btreeGetPage(pBt, pgno, ref ppPage, 0);
                if (rc == SQLITE.OK)
                {
                    rc = btreeInitPage(ppPage);
                    if (rc != SQLITE.OK)
                        releasePage(ppPage);
                }
            }
            Debug.Assert(pgno != 0 || rc == SQLITE.CORRUPT);
            return rc;
        }

        internal static void releasePage(MemPage pPage)
        {
            if (pPage != null)
            {
                Debug.Assert(pPage.aData != null);
                Debug.Assert(pPage.pBt != null);
                // TODO -- find out why corrupt9 & diskfull fail on this tests 
                //Debug.Assert( Pager.sqlite3PagerGetExtra( pPage.pDbPage ) == pPage );
                //Debug.Assert( Pager.sqlite3PagerGetData( pPage.pDbPage ) == pPage.aData );
                Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
                Pager.sqlite3PagerUnref(pPage.pDbPage);
            }
        }

#if DEBUG
        internal static void assertParentIndex(MemPage pParent, int iIdx, Pgno iChild)
        {
            Debug.Assert(iIdx <= pParent.nCell);
            if (iIdx == pParent.nCell)
                Debug.Assert(ConvertEx.sqlite3Get4byte(pParent.aData, pParent.hdrOffset + 8) == iChild);
            else
                Debug.Assert(ConvertEx.sqlite3Get4byte(pParent.aData, findCell(pParent, iIdx)) == iChild);
        }
#else
        internal static void assertParentIndex(MemPage pParent, int iIdx, Pgno iChild) { }
#endif

        internal static SQLITE fillInCell(MemPage pPage, byte[] pCell, byte[] pKey, long nKey, byte[] pData, int nData, int nZero, ref int pnSize)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            // pPage is not necessarily writeable since pCell might be auxiliary buffer space that is separate from the pPage buffer area
            // TODO -- Determine if the following Assert is needed under c#
            //Debug.Assert( pCell < pPage.aData || pCell >= &pPage.aData[pBt.pageSize] || sqlite3PagerIswriteable(pPage.pDbPage) );
            // Fill in the header.
            var nHeader = 0;
            if (pPage.leaf == 0)
                nHeader += 4;
            if (pPage.hasData != 0)
                nHeader += (int)ConvertEx.putVarint(pCell, nHeader, (int)(nData + nZero));
            else
                nData = nZero = 0;
            nHeader += ConvertEx.putVarint(pCell, nHeader, (ulong)nKey);
            var info = new CellInfo();
            btreeParseCellPtr(pPage, pCell, ref info);
            Debug.Assert(info.nHeader == nHeader);
            Debug.Assert(info.nKey == nKey);
            Debug.Assert(info.nData == (uint)(nData + nZero));
            // Fill in the payload
            var nPayload = nData + nZero;
            byte[] pSrc;
            int nSrc;
            if (pPage.intKey != 0)
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
            var pBt = pPage.pBt;
            Pgno pgnoOvfl = 0;
            MemPage pToRelease = null;
            while (nPayload > 0)
            {
                if (spaceLeft == 0)
                {
#if !SQLITE_OMIT_AUTOVACUUM
                    var pgnoPtrmap = pgnoOvfl; // Overflow page pointer-map entry page
                    if (pBt.autoVacuum)
                    {
                        do
                            pgnoOvfl++;
                        while (PTRMAP_ISPAGE(pBt, pgnoOvfl) || pgnoOvfl == PENDING_BYTE_PAGE(pBt));
                    }
#endif
                    MemPage pOvfl = null;
                    var rc = allocateBtreePage(pBt, ref pOvfl, ref pgnoOvfl, pgnoOvfl, 0);
#if !SQLITE_OMIT_AUTOVACUUM
                    // If the database supports auto-vacuum, and the second or subsequent overflow page is being allocated, add an entry to the pointer-map for that page now.
                    // If this is the first overflow page, then write a partial entry to the pointer-map. If we write nothing to this pointer-map slot,
                    // then the optimistic overflow chain processing in clearCell() may misinterpret the uninitialised values and delete the
                    // wrong pages from the database.
                    if (pBt.autoVacuum && rc == SQLITE.OK)
                    {
                        var eType = (byte)(pgnoPtrmap != 0 ? PTRMAP_OVERFLOW2 : PTRMAP_OVERFLOW1);
                        ptrmapPut(pBt, pgnoOvfl, eType, pgnoPtrmap, ref rc);
                        if (rc != SQLITE.OK)
                            releasePage(pOvfl);
                    }
#endif
                    if (rc != SQLITE.OK)
                    {
                        releasePage(pToRelease);
                        return rc;
                    }
                    // If pToRelease is not zero than pPrior points into the data area of pToRelease.  Make sure pToRelease is still writeable.
                    Debug.Assert(pToRelease == null || Pager.sqlite3PagerIswriteable(pToRelease.pDbPage));
                    // If pPrior is part of the data area of pPage, then make sure pPage is still writeable
                    // TODO -- Determine if the following Assert is needed under c#
                    //Debug.Assert( pPrior < pPage.aData || pPrior >= &pPage.aData[pBt.pageSize] || sqlite3PagerIswriteable(pPage.pDbPage) );
                    ConvertEx.sqlite3Put4byte(pPrior, pPriorIndex, pgnoOvfl);
                    releasePage(pToRelease);
                    pToRelease = pOvfl;
                    pPrior = pOvfl.aData;
                    pPriorIndex = 0;
                    ConvertEx.sqlite3Put4byte(pPrior, 0);
                    pPayload = pOvfl.aData;
                    pPayloadIndex = 4;
                    spaceLeft = (int)pBt.usableSize - 4;
                }
                var n = nPayload;
                if (n > spaceLeft)
                    n = spaceLeft;
                // If pToRelease is not zero than pPayload points into the data area of pToRelease.  Make sure pToRelease is still writeable.
                Debug.Assert(pToRelease == null || Pager.sqlite3PagerIswriteable(pToRelease.pDbPage));
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
            releasePage(pToRelease);
            return SQLITE.OK;
        }

        internal static void dropCell(MemPage pPage, int idx, int sz, ref SQLITE pRC)
        {
            if (pRC != SQLITE.OK)
                return;
            Debug.Assert(idx >= 0 && idx < pPage.nCell);
#if DEBUG
            Debug.Assert(sz == cellSize(pPage, idx));
#endif
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            var data = pPage.aData;
            var ptr = pPage.cellOffset + 2 * idx; // Used to move bytes around within data[]
            var pc = (uint)ConvertEx.get2byte(data, ptr); // Offset to cell content of cell being deleted
            var hdr = pPage.hdrOffset; // Beginning of the header.  0 most pages.  100 page 1
            if (pc < (uint)ConvertEx.get2byte(data, hdr + 5) || pc + sz > pPage.pBt.usableSize)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                return;
            }
            var rc = freeSpace(pPage, pc, sz);
            if (rc != SQLITE.OK)
            {
                pRC = rc;
                return;
            }
            Buffer.BlockCopy(data, ptr + 2, data, ptr, (pPage.nCell - 1 - idx) * 2);
            pPage.nCell--;
            data[pPage.hdrOffset + 3] = (byte)(pPage.nCell >> 8);
            data[pPage.hdrOffset + 4] = (byte)(pPage.nCell);
            pPage.nFree += 2;
        }

        internal static void insertCell(MemPage pPage, int i, byte[] pCell, int sz, byte[] pTemp, Pgno iChild, ref SQLITE pRC)
        {
            var nSkip = (iChild != 0 ? 4 : 0);
            if (pRC != SQLITE.OK)
                return;
            Debug.Assert(i >= 0 && i <= pPage.nCell + pPage.nOverflow);
            Debug.Assert(pPage.nCell <= MX_CELL(pPage.pBt) && MX_CELL(pPage.pBt) <= 10921);
            Debug.Assert(pPage.nOverflow <= pPage.aOvfl.Length);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            // The cell should normally be sized correctly.  However, when moving a malformed cell from a leaf page to an interior page, if the cell size
            // wanted to be less than 4 but got rounded up to 4 on the leaf, then size might be less than 8 (leaf-size + pointer) on the interior node.  Hence
            // the term after the || in the following assert().
            Debug.Assert(sz == cellSizePtr(pPage, pCell) || (sz == 8 && iChild > 0));
            if (pPage.nOverflow != 0 || sz + 2 > pPage.nFree)
            {
                if (pTemp != null)
                {
                    Buffer.BlockCopy(pCell, nSkip, pTemp, nSkip, sz - nSkip);
                    pCell = pTemp;
                }
                if (iChild != 0)
                    ConvertEx.sqlite3Put4byte(pCell, iChild);
                var j = pPage.nOverflow++;
                Debug.Assert(j < pPage.aOvfl.Length);
                pPage.aOvfl[j].pCell = pCell;
                pPage.aOvfl[j].idx = (ushort)i;
            }
            else
            {
                var rc = Pager.sqlite3PagerWrite(pPage.pDbPage);
                if (rc != SQLITE.OK)
                {
                    pRC = rc;
                    return;
                }
                Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
                var data = pPage.aData; // The content of the whole page
                var cellOffset = (int)pPage.cellOffset; // Address of first cell pointer in data[]
                var end = cellOffset + 2 * pPage.nCell; // First byte past the last cell pointer in data[]
                var ins = cellOffset + 2 * i; // Index in data[] where new cell pointer is inserted
                int idx = 0; // Where to write new cell content in data[]
                rc = allocateSpace(pPage, sz, ref idx);
                if (rc != SQLITE.OK)
                {
                    pRC = rc;
                    return;
                }
                // The allocateSpace() routine guarantees the following two properties if it returns success
                Debug.Assert(idx >= end + 2);
                Debug.Assert(idx + sz <= (int)pPage.pBt.usableSize);
                pPage.nCell++;
                pPage.nFree -= (ushort)(2 + sz);
                Buffer.BlockCopy(pCell, nSkip, data, idx + nSkip, sz - nSkip);
                if (iChild != 0)
                    ConvertEx.sqlite3Put4byte(data, idx, iChild);
                for (var j = end; j > ins; j -= 2)
                {
                    data[j + 0] = data[j - 2];
                    data[j + 1] = data[j - 1];
                }
                ConvertEx.put2byte(data, ins, idx);
                ConvertEx.put2byte(data, pPage.hdrOffset + 3, pPage.nCell);
#if !SQLITE_OMIT_AUTOVACUUM
                if (pPage.pBt.autoVacuum)
                    // The cell may contain a pointer to an overflow page. If so, write the entry for the overflow page into the pointer map.
                    ptrmapPutOvflPtr(pPage, pCell, ref pRC);
#endif
            }
        }


        internal static void freePage(MemPage pPage, ref SQLITE pRC)
        {
            if (pRC == SQLITE.OK)
                pRC = BtShared.freePage2(pPage.pBt, pPage, pPage.pgno);
        }

        internal static SQLITE clearCell(MemPage pPage, int pCell)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            var info = new CellInfo();
            btreeParseCellPtr(pPage, pCell, ref info);
            if (info.iOverflow == 0)
                return SQLITE.OK;  // No overflow pages. Return without doing anything
            var pBt = pPage.pBt;
            var ovflPgno = (Pgno)ConvertEx.sqlite3Get4byte(pPage.aData, pCell, info.iOverflow);
            Debug.Assert(pBt.usableSize > 4);
            var ovflPageSize = (uint)(pBt.usableSize - 4);
            var nOvfl = (int)((info.nPayload - info.nLocal + ovflPageSize - 1) / ovflPageSize);
            Debug.Assert(ovflPgno == 0 || nOvfl > 0);
            SQLITE rc;
            while (nOvfl-- != 0)
            {
                Pgno iNext = 0;
                MemPage pOvfl = null;
                if (ovflPgno < 2 || ovflPgno > btreePagecount(pBt))
                    // 0 is not a legal page number and page 1 cannot be an overflow page. Therefore if ovflPgno<2 or past the end of the file the database must be corrupt.
                    return SysEx.SQLITE_CORRUPT_BKPT();
                if (nOvfl != 0)
                {
                    rc = BtShared.getOverflowPage(pBt, ovflPgno, out pOvfl, out iNext);
                    if (rc != SQLITE.OK)
                        return rc;
                }
                if ((pOvfl != null || ((pOvfl = btreePageLookup(pBt, ovflPgno)) != null)) && Pager.sqlite3PagerPageRefcount(pOvfl.pDbPage) != 1)
                    // There is no reason any cursor should have an outstanding reference to an overflow page belonging to a cell that is being deleted/updated.
                    // So if there exists more than one reference to this page, then it must not really be an overflow page and the database must be corrupt. 
                    // It is helpful to detect this before calling freePage2(), as freePage2() may zero the page contents if secure-delete mode is
                    // enabled. If this 'overflow' page happens to be a page that the caller is iterating through or using in some other way, this
                    // can be problematic.
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                else
                    rc = BtShared.freePage2(pBt, pOvfl, ovflPgno);
                if (pOvfl != null)
                    Pager.sqlite3PagerUnref(pOvfl.pDbPage);
                if (rc != SQLITE.OK)
                    return rc;
                ovflPgno = iNext;
            }
            return SQLITE.OK;
        }
    }
}
