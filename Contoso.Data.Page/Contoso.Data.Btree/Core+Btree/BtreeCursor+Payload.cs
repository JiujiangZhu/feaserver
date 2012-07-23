using System;
using System.Diagnostics;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class BtreeCursor
    {
        // was:fetchPayload
        private static RC CopyPayload(DbPage dbPage, byte[] payload, uint payloadOffset, byte[] b, uint bOffset, uint count, bool writeOperation)
        {
            if (writeOperation)
            {
                // Copy data from buffer to page (a write operation)
                var rc = Pager.Write(dbPage);
                if (rc != RC.OK)
                    return rc;
                Buffer.BlockCopy(b, (int)bOffset, payload, (int)payloadOffset, (int)count);
            }
            else
                // Copy data from page to buffer (a read operation)
                Buffer.BlockCopy(payload, (int)payloadOffset, b, (int)bOffset, (int)count);
            return RC.OK;
        }

        // was:fetchPayload
        private RC AccessPayload(uint offset, uint size, byte[] b, bool writeOperation)
        {
            var page = Pages[PageID];   // Btree page of current entry
            Debug.Assert(page != null);
            Debug.Assert(State == CursorState.VALID);
            Debug.Assert(PagesIndexs[PageID] < page.Cells);
            Debug.Assert(HoldsMutex());
            GetCellInfo();
            var payload = Info.Cells;
            var nKey = (uint)(page.HasIntKey ? 0 : (int)Info.nKey);
            var shared = Shared; // Btree this cursor belongs to
            if (Check.NEVER(offset + size > nKey + Info.nData) || Info.nLocal > shared.UsableSize)
                // Trying to read or write past the end of the data is an error
                return SysEx.SQLITE_CORRUPT_BKPT();
            // Check if data must be read/written to/from the btree page itself.
            var rc = RC.OK;
            uint bOffset = 0;
            if (offset < Info.nLocal)
            {
                var a = (int)size;
                if (a + offset > Info.nLocal)
                    a = (int)(Info.nLocal - offset);
                rc = CopyPayload(page.DbPage, payload, (uint)(offset + Info.CellID + Info.nHeader), b, bOffset, (uint)a, writeOperation);
                offset = 0;
                bOffset += (uint)a;
                size -= (uint)a;
            }
            else
                offset -= Info.nLocal;
            var iIdx = 0;
            if (rc == RC.OK && size > 0)
            {
                var ovflSize = (uint)(shared.UsableSize - 4);  // Bytes content per ovfl page
                var nextPage = (Pgno)ConvertEx.Get4(payload, Info.nLocal + Info.CellID + Info.nHeader);
#if !SQLITE_OMIT_INCRBLOB
                // If the isIncrblobHandle flag is set and the BtCursor.aOverflow[] has not been allocated, allocate it now. The array is sized at
                // one entry for each overflow page in the overflow chain. The page number of the first overflow page is stored in aOverflow[0],
                // etc. A value of 0 in the aOverflow[] array means "not yet known" (the cache is lazily populated).
                if (IsIncrblob && OverflowIDs == null)
                {
                    var nOvfl = (Info.nPayload - Info.nLocal + ovflSize - 1) / ovflSize;
                    OverflowIDs = new Pgno[nOvfl];
                }
                // If the overflow page-list cache has been allocated and the entry for the first required overflow page is valid, skip directly to it.
                if (OverflowIDs != null && OverflowIDs[offset / ovflSize] != 0)
                {
                    iIdx = (int)(offset / ovflSize);
                    nextPage = OverflowIDs[iIdx];
                    offset = (offset % ovflSize);
                }
#endif
                for (; rc == RC.OK && size > 0 && nextPage != 0; iIdx++)
                {
#if !SQLITE_OMIT_INCRBLOB
                    // If required, populate the overflow page-list cache.
                    if (OverflowIDs != null)
                    {
                        Debug.Assert(OverflowIDs[iIdx] == 0 || OverflowIDs[iIdx] == nextPage);
                        OverflowIDs[iIdx] = nextPage;
                    }
#endif
                    MemPage MemPageDummy = null;
                    if (offset >= ovflSize)
                    {
                        // The only reason to read this page is to obtain the page number for the next page in the overflow chain. The page
                        // data is not required. So first try to lookup the overflow page-list cache, if any, then fall back to the getOverflowPage() function.
#if !SQLITE_OMIT_INCRBLOB
                        if (OverflowIDs != null && OverflowIDs[iIdx + 1] != 0)
                            nextPage = OverflowIDs[iIdx + 1];
                        else
#endif
                            rc = shared.getOverflowPage(nextPage, out MemPageDummy, out nextPage);
                        offset -= ovflSize;
                    }
                    else
                    {
                        // Need to read this page properly. It contains some of the range of data that is being read (eOp==null) or written (eOp!=null).
                        var pDbPage = new PgHdr();
                        var a = (int)size;
                        rc = shared.Pager.Get(nextPage, ref pDbPage);
                        if (rc == RC.OK)
                        {
                            payload = Pager.sqlite3PagerGetData(pDbPage);
                            nextPage = ConvertEx.Get4(payload);
                            if (a + offset > ovflSize)
                                a = (int)(ovflSize - offset);
                            rc = CopyPayload(pDbPage, payload, offset + 4, b, bOffset, (uint)a, writeOperation);
                            Pager.Unref(pDbPage);
                            offset = 0;
                            size -= (uint)a;
                            bOffset += (uint)a;
                        }
                    }
                }
            }
            if (rc == RC.OK && size > 0)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return rc;
        }

        // was:fetchPayload
        private byte[] FetchPayload(ref int size, out int offset, bool skipKey)
        {
            Debug.Assert(this != null && PageID >= 0 && Pages[PageID] != null);
            Debug.Assert(State == CursorState.VALID);
            Debug.Assert(HoldsMutex());
            offset = -1;
            var page = Pages[PageID];
            Debug.Assert(PagesIndexs[PageID] < page.Cells);
            if (Check.NEVER(Info.nSize == 0))
                Pages[PageID].ParseCell(PagesIndexs[PageID], ref Info);
            var payload = MallocEx.sqlite3Malloc(Info.nSize - Info.nHeader);
            var nKey = (page.HasIntKey ? 0 : (uint)Info.nKey);
            uint nLocal;
            if (skipKey)
            {
                offset = (int)(Info.CellID + Info.nHeader + nKey);
                Buffer.BlockCopy(Info.Cells, offset, payload, 0, (int)(Info.nSize - Info.nHeader - nKey));
                nLocal = Info.nLocal - nKey;
            }
            else
            {
                offset = (int)(Info.CellID + Info.nHeader);
                Buffer.BlockCopy(Info.Cells, offset, payload, 0, Info.nSize - Info.nHeader);
                nLocal = Info.nLocal;
                Debug.Assert(nLocal <= nKey);
            }
            size = (int)nLocal;
            return payload;
        }
    }
}
