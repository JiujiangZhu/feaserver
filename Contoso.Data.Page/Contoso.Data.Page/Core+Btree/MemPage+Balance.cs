using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;

namespace Contoso.Core
{
    public partial class MemPage
    {
        static int NN = 1;              // Number of neighbors on either side of pPage
        static int NB = (NN * 2 + 1);   // Total pages involved in the balance

#if !SQLITE_OMIT_QUICKBALANCE
        internal static SQLITE balance_quick(MemPage pParent, MemPage pPage, byte[] pSpace)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(Pager.sqlite3PagerIswriteable(pParent.pDbPage));
            Debug.Assert(pPage.nOverflow == 1);
            // This error condition is now caught prior to reaching this function
            if (pPage.nCell <= 0)
                return SysEx.SQLITE_CORRUPT_BKPT();
            // Allocate a new page. This page will become the right-sibling of pPage. Make the parent page writable, so that the new divider cell
            // may be inserted. If both these operations are successful, proceed.
            var pBt = pPage.pBt; // B-Tree Database
            var pNew = new MemPage(); // Newly allocated page
            Pgno pgnoNew = 0; // Page number of pNew
            var rc = pBt.allocateBtreePage(ref pNew, ref pgnoNew, 0, 0);
            if (rc != SQLITE.OK)
                return rc;
            var pOut = 4;
            var pCell = pPage.aOvfl[0].pCell;
            var szCell = new int[1] { pPage.cellSizePtr(pCell) };
            Debug.Assert(Pager.sqlite3PagerIswriteable(pNew.pDbPage));
            Debug.Assert(pPage.aData[0] == (Btree.PTF_INTKEY | Btree.PTF_LEAFDATA | Btree.PTF_LEAF));
            pNew.zeroPage(Btree.PTF_INTKEY | Btree.PTF_LEAFDATA | Btree.PTF_LEAF);
            pNew.assemblePage(1, pCell, szCell);
            // If this is an auto-vacuum database, update the pointer map with entries for the new page, and any pointer from the
            // cell on the page to an overflow page. If either of these operations fails, the return code is set, but the contents
            // of the parent page are still manipulated by thh code below. That is Ok, at this point the parent page is guaranteed to
            // be marked as dirty. Returning an error code will cause a rollback, undoing any changes made to the parent page.
#if !SQLITE_OMIT_AUTOVACUUM
            if (pBt.autoVacuum)
#else
if (false)
#endif
            {
                pBt.ptrmapPut(pgnoNew, PTRMAP.BTREE, pParent.pgno, ref rc);
                if (szCell[0] > pNew.minLocal)
                    pNew.ptrmapPutOvflPtr(pCell, ref rc);
            }
            // Create a divider cell to insert into pParent. The divider cell consists of a 4-byte page number (the page number of pPage) and
            // a variable length key value (which must be the same value as the largest key on pPage).
            // To find the largest key value on pPage, first find the right-most cell on pPage. The first two fields of this cell are the
            // record-length (a variable length integer at most 32-bits in size) and the key value (a variable length integer, may have any value).
            // The first of the while(...) loops below skips over the record-length field. The second while(...) loop copies the key value from the
            // cell on pPage into the pSpace buffer.
            var iCell = pPage.findCell(pPage.nCell - 1);
            pCell = pPage.aData;
            var _pCell = iCell;
            var pStop = _pCell + 9;
            while (((pCell[_pCell++]) & 0x80) != 0 && _pCell < pStop) ;
            pStop = _pCell + 9;
            while (((pSpace[pOut++] = pCell[_pCell++]) & 0x80) != 0 && _pCell < pStop) ;
            // Insert the new divider cell into pParent.
            pParent.insertCell(pParent.nCell, pSpace, pOut, null, pPage.pgno, ref rc);
            // Set the right-child pointer of pParent to point to the new page.
            ConvertEx.sqlite3Put4byte(pParent.aData, pParent.hdrOffset + 8, pgnoNew);
            // Release the reference to the new page.
            pNew.releasePage();
            return rc;
        }
#endif

        internal void assemblePage(int nCell, byte[] apCell, int[] aSize)
        {
            Debug.Assert(this.nOverflow == 0);
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBt.mutex));
            Debug.Assert(nCell >= 0 && nCell <= (int)Btree.MX_CELL(this.pBt) && (int)Btree.MX_CELL(this.pBt) <= 10921);
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.pDbPage));
            // Check that the page has just been zeroed by zeroPage()
            Debug.Assert(this.nCell == 0);
            //
            var data = this.aData; // Pointer to data for pPage
            int hdr = this.hdrOffset; // Offset of header on pPage
            var nUsable = (int)this.pBt.usableSize; // Usable size of page
            Debug.Assert(ConvertEx.get2byteNotZero(data, hdr + 5) == nUsable);
            var pCellptr = this.cellOffset + nCell * 2; // Address of next cell pointer
            var cellbody = nUsable; // Address of next cell body
            for (var i = nCell - 1; i >= 0; i--)
            {
                var sz = (ushort)aSize[i];
                pCellptr -= 2;
                cellbody -= sz;
                ConvertEx.put2byte(data, pCellptr, cellbody);
                Buffer.BlockCopy(apCell, 0, data, cellbody, sz);
            }
            ConvertEx.put2byte(data, hdr + 3, nCell);
            ConvertEx.put2byte(data, hdr + 5, cellbody);
            this.nFree -= (ushort)(nCell * 2 + nUsable - cellbody);
            this.nCell = (ushort)nCell;
        }
        internal void assemblePage(int nCell, byte[][] apCell, ushort[] aSize, int offset)
        {
            Debug.Assert(this.nOverflow == 0);
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBt.mutex));
            Debug.Assert(nCell >= 0 && nCell <= Btree.MX_CELL(this.pBt) && Btree.MX_CELL(this.pBt) <= 5460);
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.pDbPage));
            // Check that the page has just been zeroed by zeroPage()
            Debug.Assert(this.nCell == 0);
            //
            var data = this.aData; // Pointer to data for pPage
            var hdr = this.hdrOffset; // Offset of header on pPage
            var nUsable = (int)this.pBt.usableSize; // Usable size of page
            Debug.Assert(ConvertEx.get2byte(data, hdr + 5) == nUsable);
            var pCellptr = this.cellOffset + nCell * 2; // Address of next cell pointer
            var cellbody = nUsable; // Address of next cell body
            for (var i = nCell - 1; i >= 0; i--)
            {
                pCellptr -= 2;
                cellbody -= aSize[i + offset];
                ConvertEx.put2byte(data, pCellptr, cellbody);
                Buffer.BlockCopy(apCell[offset + i], 0, data, cellbody, aSize[i + offset]);
            }
            ConvertEx.put2byte(data, hdr + 3, nCell);
            ConvertEx.put2byte(data, hdr + 5, cellbody);
            this.nFree -= (ushort)(nCell * 2 + nUsable - cellbody);
            this.nCell = (ushort)nCell;
        }
        internal void assemblePage(int nCell, byte[] apCell, ushort[] aSize)
        {
            Debug.Assert(this.nOverflow == 0);
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBt.mutex));
            Debug.Assert(nCell >= 0 && nCell <= Btree.MX_CELL(this.pBt) && Btree.MX_CELL(this.pBt) <= 5460);
            Debug.Assert(Pager.sqlite3PagerIswriteable(this.pDbPage));
            // Check that the page has just been zeroed by zeroPage()
            Debug.Assert(this.nCell == 0);
            //
            var data = this.aData; // Pointer to data for pPage
            var hdr = this.hdrOffset; // Offset of header on pPage
            var nUsable = (int)this.pBt.usableSize; // Usable size of page
            Debug.Assert(ConvertEx.get2byte(data, hdr + 5) == nUsable);
            var pCellptr = this.cellOffset + nCell * 2; // Address of next cell pointer
            var cellbody = nUsable; // Address of next cell body
            for (var i = nCell - 1; i >= 0; i--)
            {
                pCellptr -= 2;
                cellbody -= aSize[i];
                ConvertEx.put2byte(data, pCellptr, cellbody);
                Buffer.BlockCopy(apCell, 0, data, cellbody, aSize[i]);
            }
            ConvertEx.put2byte(data, hdr + 3, nCell);
            ConvertEx.put2byte(data, hdr + 5, cellbody);
            this.nFree -= (ushort)(nCell * 2 + nUsable - cellbody);
            this.nCell = (ushort)nCell;
        }

        internal static void copyNodeContent(MemPage pFrom, MemPage pTo, ref SQLITE pRC)
        {
            if (pRC != SQLITE.OK)
                return;
            var pBt = pFrom.pBt;
            var aFrom = pFrom.aData;
            var aTo = pTo.aData;
            var iFromHdr = pFrom.hdrOffset;
            var iToHdr = (pTo.pgno == 1 ? 100 : 0);
            Debug.Assert(pFrom.isInit != 0);
            Debug.Assert(pFrom.nFree >= iToHdr);
            Debug.Assert(ConvertEx.get2byte(aFrom, iFromHdr + 5) <= (int)pBt.usableSize);
            // Copy the b-tree node content from page pFrom to page pTo.
            var iData = (int)ConvertEx.get2byte(aFrom, iFromHdr + 5);
            Buffer.BlockCopy(aFrom, iData, aTo, iData, (int)pBt.usableSize - iData);
            Buffer.BlockCopy(aFrom, iFromHdr, aTo, iToHdr, pFrom.cellOffset + 2 * pFrom.nCell);
            // Reinitialize page pTo so that the contents of the MemPage structure match the new data. The initialization of pTo can actually fail under
            // fairly obscure circumstances, even though it is a copy of initialized  page pFrom.
            pTo.isInit = 0;
            var rc = pTo.btreeInitPage();
            if (rc != SQLITE.OK)
            {
                pRC = rc;
                return;
            }
            // If this is an auto-vacuum database, update the pointer-map entries for any b-tree or overflow pages that pTo now contains the pointers to.
#if !SQLITE_OMIT_AUTOVACUUM
            if (pBt.autoVacuum)
#else
if (false)
#endif
            {
                pRC = pTo.setChildPtrmaps();
            }
        }

        internal static SQLITE balance_nonroot(MemPage pParent, int iParentIdx, byte[] aOvflSpace, int isRoot)
        {
            var apOld = new MemPage[NB];    // pPage and up to two siblings
            var apCopy = new MemPage[NB];   // Private copies of apOld[] pages
            var apNew = new MemPage[NB + 2];// pPage and up to NB siblings after balancing
            var apDiv = new int[NB - 1];        // Divider cells in pParent
            var cntNew = new int[NB + 2];       // Index in aCell[] of cell after i-th page
            var szNew = new int[NB + 2];        // Combined size of cells place on i-th page
            var szCell = new ushort[1];            // Local size of all cells in apCell[]
            BtShared pBt;                // The whole database
            int nCell = 0;               // Number of cells in apCell[]
            int nMaxCells = 0;           // Allocated size of apCell, szCell, aFrom.
            int nNew = 0;                // Number of pages in apNew[]
            ushort leafCorrection;          // 4 if pPage is a leaf.  0 if not
            int leafData;                // True if pPage is a leaf of a LEAFDATA tree
            int usableSpace;             // Bytes in pPage beyond the header
            int pageFlags;               // Value of pPage.aData[0]
            int subtotal;                // Subtotal of bytes in cells on one page
            int iOvflSpace = 0;          // First unused byte of aOvflSpace[]
            int szScratch;               // Size of scratch memory requested
            byte[][] apCell = null;                 // All cells begin balanced
            //
            pBt = pParent.pBt;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(Pager.sqlite3PagerIswriteable(pParent.pDbPage));
#if false
            Btree.TRACE("BALANCE: begin page %d child of %d\n", pPage.pgno, pParent.pgno);
#endif
            // At this point pParent may have at most one overflow cell. And if this overflow cell is present, it must be the cell with
            // index iParentIdx. This scenario comes about when this function is called (indirectly) from sqlite3BtreeDelete().
            Debug.Assert(pParent.nOverflow == 0 || pParent.nOverflow == 1);
            Debug.Assert(pParent.nOverflow == 0 || pParent.aOvfl[0].idx == iParentIdx);
            // Find the sibling pages to balance. Also locate the cells in pParent that divide the siblings. An attempt is made to find NN siblings on
            // either side of pPage. More siblings are taken from one side, however, if there are fewer than NN siblings on the other side. If pParent
            // has NB or fewer children then all children of pParent are taken.
            // This loop also drops the divider cells from the parent page. This way, the remainder of the function does not have to deal with any
            // overflow cells in the parent page, since if any existed they will have already been removed.
            int nOld; // Number of pages in apOld[]
            int nxDiv; // Next divider slot in pParent.aCell[]
            var i = pParent.nOverflow + pParent.nCell;
            if (i < 2)
            {
                nxDiv = 0;
                nOld = i + 1;
            }
            else
            {
                nOld = 3;
                if (iParentIdx == 0)
                    nxDiv = 0;
                else if (iParentIdx == i)
                    nxDiv = i - 2;
                else
                    nxDiv = iParentIdx - 1;
                i = 2;
            }
            var pRight = ((i + nxDiv - pParent.nOverflow) == pParent.nCell ? pParent.hdrOffset + 8 : pParent.findCell(i + nxDiv - pParent.nOverflow)); // Location in parent of right-sibling pointer
            var pgno = (Pgno)ConvertEx.sqlite3Get4byte(pParent.aData, pRight);
            var rc = SQLITE.OK;
            while (true)
            {
                rc = pBt.getAndInitPage(pgno, ref apOld[i]);
                if (rc != SQLITE.OK)
                    goto balance_cleanup;
                nMaxCells += 1 + apOld[i].nCell + apOld[i].nOverflow;
                if (i-- == 0)
                    break;
                if (i + nxDiv == pParent.aOvfl[0].idx && pParent.nOverflow != 0)
                {
                    apDiv[i] = 0;
                    pgno = ConvertEx.sqlite3Get4byte(pParent.aOvfl[0].pCell, apDiv[i]);
                    szNew[i] = pParent.cellSizePtr(apDiv[i]);
                    pParent.nOverflow = 0;
                }
                else
                {
                    apDiv[i] = pParent.findCell(i + nxDiv - pParent.nOverflow);
                    pgno = ConvertEx.sqlite3Get4byte(pParent.aData, apDiv[i]);
                    szNew[i] = pParent.cellSizePtr(apDiv[i]);
                    // Drop the cell from the parent page. apDiv[i] still points to the cell within the parent, even though it has been dropped.
                    // This is safe because dropping a cell only overwrites the first four bytes of it, and this function does not need the first
                    // four bytes of the divider cell. So the pointer is safe to use later on.
                    //
                    // Unless SQLite is compiled in secure-delete mode. In this case, the dropCell() routine will overwrite the entire cell with zeroes.
                    // In this case, temporarily copy the cell into the aOvflSpace[] buffer. It will be copied out again as soon as the aSpace[] buffer
                    // is allocated.
                    //if (pBt.secureDelete)
                    //{
                    //  int iOff = (int)(apDiv[i]) - (int)(pParent.aData); //SQLITE_PTR_TO_INT(apDiv[i]) - SQLITE_PTR_TO_INT(pParent.aData);
                    //         if( (iOff+szNew[i])>(int)pBt->usableSize )
                    //  {
                    //    rc = SQLITE_CORRUPT_BKPT();
                    //    Array.Clear(apOld[0].aData,0,apOld[0].aData.Length); //memset(apOld, 0, (i + 1) * sizeof(MemPage*));
                    //    goto balance_cleanup;
                    //  }
                    //  else
                    //  {
                    //    memcpy(&aOvflSpace[iOff], apDiv[i], szNew[i]);
                    //    apDiv[i] = &aOvflSpace[apDiv[i] - pParent.aData];
                    //  }
                    //}
                    pParent.dropCell(i + nxDiv - pParent.nOverflow, szNew[i], ref rc);
                }
            }
            // Make nMaxCells a multiple of 4 in order to preserve 8-byte alignment
            nMaxCells = (nMaxCells + 3) & ~3;
            // Allocate space for memory structures
            apCell = MallocEx.sqlite3ScratchMalloc(apCell, nMaxCells);
            if (szCell.Length < nMaxCells)
                Array.Resize(ref szCell, nMaxCells);
            // Load pointers to all cells on sibling pages and the divider cells into the local apCell[] array.  Make copies of the divider cells
            // into space obtained from aSpace1[] and remove the the divider Cells from pParent.
            // If the siblings are on leaf pages, then the child pointers of the divider cells are stripped from the cells before they are copied
            // into aSpace1[].  In this way, all cells in apCell[] are without child pointers.  If siblings are not leaves, then all cell in
            // apCell[] include child pointers.  Either way, all cells in apCell[] are alike.
            // leafCorrection:  4 if pPage is a leaf.  0 if pPage is not a leaf.
            //       leafData:  1 if pPage holds key+data and pParent holds only keys.
            leafCorrection = (ushort)(apOld[0].leaf * 4);
            leafData = apOld[0].hasData;
            int j;
            for (i = 0; i < nOld; i++)
            {
                // Before doing anything else, take a copy of the i'th original sibling The rest of this function will use data from the copies rather
                // that the original pages since the original pages will be in the process of being overwritten.
                var pOld = apCopy[i] = apOld[i].Copy();
                var limit = pOld.nCell + pOld.nOverflow;
                if (pOld.nOverflow > 0 || true)
                {
                    for (j = 0; j < limit; j++)
                    {
                        Debug.Assert(nCell < nMaxCells);
                        var iFOFC = pOld.findOverflowCell(j);
                        szCell[nCell] = pOld.cellSizePtr(iFOFC);
                        // Copy the Data Locally
                        if (apCell[nCell] == null)
                            apCell[nCell] = new byte[szCell[nCell]];
                        else if (apCell[nCell].Length < szCell[nCell])
                            Array.Resize(ref apCell[nCell], szCell[nCell]);
                        if (iFOFC < 0)  // Overflow Cell
                            Buffer.BlockCopy(pOld.aOvfl[-(iFOFC + 1)].pCell, 0, apCell[nCell], 0, szCell[nCell]);
                        else
                            Buffer.BlockCopy(pOld.aData, iFOFC, apCell[nCell], 0, szCell[nCell]);
                        nCell++;
                    }
                }
                else
                {
                    var aData = pOld.aData;
                    var maskPage = pOld.maskPage;
                    var cellOffset = pOld.cellOffset;
                    for (j = 0; j < limit; j++)
                    {
                        Debugger.Break();
                        Debug.Assert(nCell < nMaxCells);
                        apCell[nCell] = findCellv2(aData, maskPage, cellOffset, j);
                        szCell[nCell] = pOld.cellSizePtr(apCell[nCell]);
                        nCell++;
                    }
                }
                if (i < nOld - 1 && 0 == leafData)
                {
                    var sz = (ushort)szNew[i];
                    var pTemp = MallocEx.sqlite3Malloc(sz + leafCorrection);
                    Debug.Assert(nCell < nMaxCells);
                    szCell[nCell] = sz;
                    Debug.Assert(sz <= pBt.maxLocal + 23);
                    Buffer.BlockCopy(pParent.aData, apDiv[i], pTemp, 0, sz);
                    if (apCell[nCell] == null || apCell[nCell].Length < sz)
                        Array.Resize(ref apCell[nCell], sz);
                    Buffer.BlockCopy(pTemp, leafCorrection, apCell[nCell], 0, sz);
                    Debug.Assert(leafCorrection == 0 || leafCorrection == 4);
                    szCell[nCell] = (ushort)(szCell[nCell] - leafCorrection);
                    if (0 == pOld.leaf)
                    {
                        Debug.Assert(leafCorrection == 0);
                        Debug.Assert(pOld.hdrOffset == 0);
                        // The right pointer of the child page pOld becomes the left pointer of the divider cell
                        Buffer.BlockCopy(pOld.aData, 8, apCell[nCell], 0, 4);//memcpy( apCell[nCell], ref pOld.aData[8], 4 );
                    }
                    else
                    {
                        Debug.Assert(leafCorrection == 4);
                        if (szCell[nCell] < 4)
                            // Do not allow any cells smaller than 4 bytes.
                            szCell[nCell] = 4;
                    }
                    nCell++;
                }
            }
            // Figure out the number of pages needed to hold all nCell cells. Store this number in "k".  Also compute szNew[] which is the total
            // size of all cells on the i-th page and cntNew[] which is the index in apCell[] of the cell that divides page i from page i+1.
            // cntNew[k] should equal nCell.
            // Values computed by this block:
            //           k: The total number of sibling pages
            //    szNew[i]: Spaced used on the i-th sibling page.
            //   cntNew[i]: Index in apCell[] and szCell[] for the first cell to
            //              the right of the i-th sibling page.
            // usableSpace: Number of bytes of space available on each sibling.
            usableSpace = (int)pBt.usableSize - 12 + leafCorrection;
            int k;
            for (subtotal = k = i = 0; i < nCell; i++)
            {
                Debug.Assert(i < nMaxCells);
                subtotal += szCell[i] + 2;
                if (subtotal > usableSpace)
                {
                    szNew[k] = subtotal - szCell[i];
                    cntNew[k] = i;
                    if (leafData != 0)
                        i--;
                    subtotal = 0;
                    k++;
                    if (k > NB + 1)
                    {
                        rc = SysEx.SQLITE_CORRUPT_BKPT();
                        goto balance_cleanup;
                    }
                }
            }
            szNew[k] = subtotal;
            cntNew[k] = nCell;
            k++;
            // The packing computed by the previous block is biased toward the siblings on the left side.  The left siblings are always nearly full, while the
            // right-most sibling might be nearly empty.  This block of code attempts to adjust the packing of siblings to get a better balance.
            //
            // This adjustment is more than an optimization.  The packing above might be so out of balance as to be illegal.  For example, the right-most
            // sibling might be completely empty.  This adjustment is not optional.
            for (i = k - 1; i > 0; i--)
            {
                var szRight = szNew[i];  // Size of sibling on the right
                var szLeft = szNew[i - 1]; // Size of sibling on the left
                var r = cntNew[i - 1] - 1; // Index of right-most cell in left sibling
                var d = r + 1 - leafData; // Index of first cell to the left of right sibling
                Debug.Assert(d < nMaxCells);
                Debug.Assert(r < nMaxCells);
                while (szRight == 0 || szRight + szCell[d] + 2 <= szLeft - (szCell[r] + 2))
                {
                    szRight += szCell[d] + 2;
                    szLeft -= szCell[r] + 2;
                    cntNew[i - 1]--;
                    r = cntNew[i - 1] - 1;
                    d = r + 1 - leafData;
                }
                szNew[i] = szRight;
                szNew[i - 1] = szLeft;
            }
            // Either we found one or more cells (cntnew[0])>0) or pPage is a virtual root page.  A virtual root page is when the real root
            // page is page 1 and we are the only child of that page.
            Debug.Assert(cntNew[0] > 0 || (pParent.pgno == 1 && pParent.nCell == 0));
            Btree.TRACE("BALANCE: old: %d %d %d  ", apOld[0].pgno, (nOld >= 2 ? apOld[1].pgno : 0), (nOld >= 3 ? apOld[2].pgno : 0));
            // Allocate k new pages.  Reuse old pages where possible.
            if (apOld[0].pgno <= 1)
            {
                rc = SysEx.SQLITE_CORRUPT_BKPT();
                goto balance_cleanup;
            }
            pageFlags = apOld[0].aData[0];
            for (i = 0; i < k; i++)
            {
                var pNew = new MemPage();
                if (i < nOld)
                {
                    pNew = apNew[i] = apOld[i];
                    apOld[i] = null;
                    rc = Pager.sqlite3PagerWrite(pNew.pDbPage);
                    nNew++;
                    if (rc != SQLITE.OK)
                        goto balance_cleanup;
                }
                else
                {
                    Debug.Assert(i > 0);
                    rc = pBt.allocateBtreePage(ref pNew, ref pgno, pgno, 0);
                    if (rc != 0)
                        goto balance_cleanup;
                    apNew[i] = pNew;
                    nNew++;

                    // Set the pointer-map entry for the new sibling page.
#if !SQLITE_OMIT_AUTOVACUUM
                    if (pBt.autoVacuum)
#else
if (false)
#endif
                    {
                        pBt.ptrmapPut(pNew.pgno, PTRMAP.BTREE, pParent.pgno, ref rc);
                        if (rc != SQLITE.OK)
                            goto balance_cleanup;
                    }
                }
            }
            // Free any old pages that were not reused as new pages.
            while (i < nOld)
            {
                apOld[i].freePage(ref rc);
                if (rc != SQLITE.OK)
                    goto balance_cleanup;
                apOld[i].releasePage();
                apOld[i] = null;
                i++;
            }
            // Put the new pages in accending order.  This helps to keep entries in the disk file in order so that a scan
            // of the table is a linear scan through the file.  That in turn helps the operating system to deliver pages
            // from the disk more rapidly.
            // An O(n^2) insertion sort algorithm is used, but since n is never more than NB (a small constant), that should
            // not be a problem.
            // When NB==3, this one optimization makes the database about 25% faster for large insertions and deletions.
            for (i = 0; i < k - 1; i++)
            {
                var minV = (int)apNew[i].pgno;
                var minI = i;
                for (j = i + 1; j < k; j++)
                    if (apNew[j].pgno < (uint)minV)
                    {
                        minI = j;
                        minV = (int)apNew[j].pgno;
                    }
                if (minI > i)
                {
                    var pT = apNew[i];
                    apNew[i] = apNew[minI];
                    apNew[minI] = pT;
                }
            }
            Btree.TRACE("new: %d(%d) %d(%d) %d(%d) %d(%d) %d(%d)\n", apNew[0].pgno, szNew[0],
                (nNew >= 2 ? apNew[1].pgno : 0), (nNew >= 2 ? szNew[1] : 0),
                (nNew >= 3 ? apNew[2].pgno : 0), (nNew >= 3 ? szNew[2] : 0),
                (nNew >= 4 ? apNew[3].pgno : 0), (nNew >= 4 ? szNew[3] : 0),
                (nNew >= 5 ? apNew[4].pgno : 0), (nNew >= 5 ? szNew[4] : 0));
            Debug.Assert(Pager.sqlite3PagerIswriteable(pParent.pDbPage));
            ConvertEx.sqlite3Put4byte(pParent.aData, pRight, apNew[nNew - 1].pgno);
            // Evenly distribute the data in apCell[] across the new pages. Insert divider cells into pParent as necessary.
            for (i = 0; i < nNew; i++)
            {
                // Assemble the new sibling page.
                MemPage pNew = apNew[i];
                Debug.Assert(j < nMaxCells);
                pNew.zeroPage(pageFlags);
                pNew.assemblePage(cntNew[i] - j, apCell, szCell, j);
                Debug.Assert(pNew.nCell > 0 || (nNew == 1 && cntNew[0] == 0));
                Debug.Assert(pNew.nOverflow == 0);
                j = cntNew[i];
                // If the sibling page assembled above was not the right-most sibling, insert a divider cell into the parent page.
                Debug.Assert(i < nNew - 1 || j == nCell);
                if (j < nCell)
                {
                    Debug.Assert(j < nMaxCells);
                    var pCell = apCell[j];
                    var sz = szCell[j] + leafCorrection;
                    var pTemp = MallocEx.sqlite3Malloc(sz);
                    if (pNew.leaf == 0)
                        Buffer.BlockCopy(pCell, 0, pNew.aData, 8, 4);
                    else if (leafData != 0)
                    {
                        // If the tree is a leaf-data tree, and the siblings are leaves, then there is no divider cell in apCell[]. Instead, the divider
                        // cell consists of the integer key for the right-most cell of the sibling-page assembled above only.
                        var info = new CellInfo();
                        j--;
                        pNew.btreeParseCellPtr(apCell[j], ref info);
                        pCell = pTemp;
                        sz = 4 + ConvertEx.putVarint(pCell, 4, (ulong)info.nKey);
                        pTemp = null;
                    }
                    else
                    {
                        //------------ pCell -= 4;
                        var _pCell_4 = MallocEx.sqlite3Malloc(pCell.Length + 4);
                        Buffer.BlockCopy(pCell, 0, _pCell_4, 4, pCell.Length);
                        pCell = _pCell_4;
                        // Obscure case for non-leaf-data trees: If the cell at pCell was previously stored on a leaf node, and its reported size was 4
                        // bytes, then it may actually be smaller than this (see btreeParseCellPtr(), 4 bytes is the minimum size of
                        // any cell). But it is important to pass the correct size to insertCell(), so reparse the cell now.
                        // Note that this can never happen in an SQLite data file, as all cells are at least 4 bytes. It only happens in b-trees used
                        // to evaluate "IN (SELECT ...)" and similar clauses.
                        if (szCell[j] == 4)
                        {
                            Debug.Assert(leafCorrection == 4);
                            sz = pParent.cellSizePtr(pCell);
                        }
                    }
                    iOvflSpace += sz;
                    Debug.Assert(sz <= pBt.maxLocal + 23);
                    Debug.Assert(iOvflSpace <= (int)pBt.pageSize);
                    pParent.insertCell(nxDiv, pCell, sz, pTemp, pNew.pgno, ref rc);
                    if (rc != SQLITE.OK)
                        goto balance_cleanup;
                    Debug.Assert(Pager.sqlite3PagerIswriteable(pParent.pDbPage));
                    j++;
                    nxDiv++;
                }
            }
            Debug.Assert(j == nCell);
            Debug.Assert(nOld > 0);
            Debug.Assert(nNew > 0);
            if ((pageFlags & Btree.PTF_LEAF) == 0)
                Buffer.BlockCopy(apCopy[nOld - 1].aData, 8, apNew[nNew - 1].aData, 8, 4);
            if (isRoot != 0 && pParent.nCell == 0 && pParent.hdrOffset <= apNew[0].nFree)
            {
                // The root page of the b-tree now contains no cells. The only sibling page is the right-child of the parent. Copy the contents of the
                // child page into the parent, decreasing the overall height of the b-tree structure by one. This is described as the "balance-shallower"
                // sub-algorithm in some documentation.
                // If this is an auto-vacuum database, the call to copyNodeContent() sets all pointer-map entries corresponding to database image pages
                // for which the pointer is stored within the content being copied.
                // The second Debug.Assert below verifies that the child page is defragmented (it must be, as it was just reconstructed using assemblePage()). This
                // is important if the parent page happens to be page 1 of the database image.  */
                Debug.Assert(nNew == 1);
                Debug.Assert(apNew[0].nFree == (ConvertEx.get2byte(apNew[0].aData, 5) - apNew[0].cellOffset - apNew[0].nCell * 2));
                copyNodeContent(apNew[0], pParent, ref rc);
                apNew[0].freePage(ref rc);
            }
            else
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum)
#else
if (false)
#endif
                {
                    // Fix the pointer-map entries for all the cells that were shifted around. There are several different types of pointer-map entries that need to
                    // be dealt with by this routine. Some of these have been set already, but many have not. The following is a summary:
                    //   1) The entries associated with new sibling pages that were not siblings when this function was called. These have already
                    //      been set. We don't need to worry about old siblings that were moved to the free-list - the freePage() code has taken care
                    //      of those.
                    //   2) The pointer-map entries associated with the first overflow page in any overflow chains used by new divider cells. These
                    //      have also already been taken care of by the insertCell() code.
                    //   3) If the sibling pages are not leaves, then the child pages of cells stored on the sibling pages may need to be updated.
                    //   4) If the sibling pages are not internal intkey nodes, then any overflow pages used by these cells may need to be updated
                    //      (internal intkey nodes never contain pointers to overflow pages).
                    //   5) If the sibling pages are not leaves, then the pointer-map entries for the right-child pages of each sibling may need
                    //      to be updated.
                    // Cases 1 and 2 are dealt with above by other code. The next block deals with cases 3 and 4 and the one after that, case 5. Since
                    // setting a pointer map entry is a relatively expensive operation, this code only sets pointer map entries for child or overflow pages that have
                    // actually moved between pages.
                    var pNew = apNew[0];
                    var pOld = apCopy[0];
                    var nOverflow = pOld.nOverflow;
                    var iNextOld = pOld.nCell + nOverflow;
                    var iOverflow = (nOverflow != 0 ? pOld.aOvfl[0].idx : -1);
                    j = 0; // Current 'old' sibling page
                    k = 0; // Current 'new' sibling page
                    for (i = 0; i < nCell; i++)
                    {
                        var isDivider = 0;
                        while (i == iNextOld)
                        {
                            // Cell i is the cell immediately following the last cell on old sibling page j. If the siblings are not leaf pages of an
                            // intkey b-tree, then cell i was a divider cell.
                            pOld = apCopy[++j];
                            iNextOld = i + (0 == leafData ? 1 : 0) + pOld.nCell + pOld.nOverflow;
                            if (pOld.nOverflow != 0)
                            {
                                nOverflow = pOld.nOverflow;
                                iOverflow = i + (0 == leafData ? 1 : 0) + pOld.aOvfl[0].idx;
                            }
                            isDivider = 0 == leafData ? 1 : 0;
                        }
                        Debug.Assert(nOverflow > 0 || iOverflow < i);
                        Debug.Assert(nOverflow < 2 || pOld.aOvfl[0].idx == pOld.aOvfl[1].idx - 1);
                        Debug.Assert(nOverflow < 3 || pOld.aOvfl[1].idx == pOld.aOvfl[2].idx - 1);
                        if (i == iOverflow)
                        {
                            isDivider = 1;
                            if (--nOverflow > 0)
                                iOverflow++;
                        }
                        if (i == cntNew[k])
                        {
                            // Cell i is the cell immediately following the last cell on new sibling page k. If the siblings are not leaf pages of an
                            // intkey b-tree, then cell i is a divider cell.
                            pNew = apNew[++k];
                            if (leafData == 0)
                                continue;
                        }
                        Debug.Assert(j < nOld);
                        Debug.Assert(k < nNew);
                        // If the cell was originally divider cell (and is not now) or an overflow cell, or if the cell was located on a different sibling
                        // page before the balancing, then the pointer map entries associated with any child or overflow pages need to be updated.
                        if (isDivider != 0 || pOld.pgno != pNew.pgno)
                        {
                            if (leafCorrection == 0)
                                pBt.ptrmapPut(ConvertEx.sqlite3Get4byte(apCell[i]), PTRMAP.BTREE, pNew.pgno, ref rc);
                            if (szCell[i] > pNew.minLocal)
                                pNew.ptrmapPutOvflPtr(apCell[i], ref rc);
                        }
                    }
                    if (leafCorrection == 0)
                        for (i = 0; i < nNew; i++)
                        {
                            var key = ConvertEx.sqlite3Get4byte(apNew[i].aData, 8);
                            pBt.ptrmapPut(key, PTRMAP.BTREE, apNew[i].pgno, ref rc);
                        }
#if false
// The ptrmapCheckPages() contains Debug.Assert() statements that verify that all pointer map pages are set correctly. This is helpful while
// debugging. This is usually disabled because a corrupt database may cause an Debug.Assert() statement to fail.
ptrmapCheckPages(apNew, nNew);
ptrmapCheckPages(pParent, 1);
#endif
                }
            Debug.Assert(pParent.isInit != 0);
            Btree.TRACE("BALANCE: finished: old=%d new=%d cells=%d\n", nOld, nNew, nCell);
        // Cleanup before returning.
        balance_cleanup:
            MallocEx.sqlite3ScratchFree(apCell);
            for (i = 0; i < nOld; i++)
                apOld[i].releasePage();
            for (i = 0; i < nNew; i++)
                apNew[i].releasePage();
            return rc;
        }

        internal static SQLITE balance_deeper(MemPage pRoot, ref MemPage ppChild)
        {
            MemPage pChild = null; // Pointer to a new child page
            Pgno pgnoChild = 0; // Page number of the new child page
            var pBt = pRoot.pBt;
            Debug.Assert(pRoot.nOverflow > 0);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            // Make pRoot, the root page of the b-tree, writable. Allocate a new page that will become the new right-child of pPage. Copy the contents
            // of the node stored on pRoot into the new child page.
            var rc = Pager.sqlite3PagerWrite(pRoot.pDbPage);
            if (rc == SQLITE.OK)
            {
                rc = pBt.allocateBtreePage(ref pChild, ref pgnoChild, pRoot.pgno, 0);
                copyNodeContent(pRoot, pChild, ref rc);
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum)
#else
if (false)
#endif
                {
                    pBt.ptrmapPut(pgnoChild, PTRMAP.BTREE, pRoot.pgno, ref rc);
                }
            }
            if (rc != SQLITE.OK)
            {
                ppChild = null;
                pChild.releasePage();
                return rc;
            }
            Debug.Assert(Pager.sqlite3PagerIswriteable(pChild.pDbPage));
            Debug.Assert(Pager.sqlite3PagerIswriteable(pRoot.pDbPage));
            Debug.Assert(pChild.nCell == pRoot.nCell);
            Btree.TRACE("BALANCE: copy root %d into %d\n", pRoot.pgno, pChild.pgno);
            // Copy the overflow cells from pRoot to pChild
            Array.Copy(pRoot.aOvfl, pChild.aOvfl, pRoot.nOverflow);
            pChild.nOverflow = pRoot.nOverflow;
            // Zero the contents of pRoot. Then install pChild as the right-child.
            pRoot.zeroPage(pChild.aData[0] & ~Btree.PTF_LEAF);
            ConvertEx.sqlite3Put4byte(pRoot.aData, pRoot.hdrOffset + 8, pgnoChild);
            ppChild = pChild;
            return SQLITE.OK;
        }
    }
}
