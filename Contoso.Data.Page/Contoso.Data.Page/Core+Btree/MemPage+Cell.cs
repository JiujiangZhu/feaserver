using System;
using System.Diagnostics;
using Contoso.Sys;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class MemPage
    {
        // was:cellSize
#if DEBUG
        internal ushort cellSize(int iCell) { return cellSizePtr(FindCell(iCell)); }
#else
        internal int cellSize(int iCell) { return -1; }
#endif

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

        // was:cellSizePtr
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
