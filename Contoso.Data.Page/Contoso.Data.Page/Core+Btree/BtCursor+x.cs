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
    public partial class BtCursor
    {
#if DEBUG
        internal static bool cursorHoldsMutex(BtCursor p) { return MutexEx.sqlite3_mutex_held(p.pBt.mutex); }
#else
        internal static bool cursorHoldsMutex(BtCursor p) { return true; }
#endif

        #region Properties

        internal static int sqlite3BtreeCursorSize() { return -1; }
        internal static void sqlite3BtreeCursorZero(BtCursor p) { p.Clear(); }

        internal static void sqlite3BtreeSetCachedRowid(BtCursor pCur, long iRowid)
        {
            for (var p = pCur.pBt.pCursor; p != null; p = p.pNext)
                if (p.pgnoRoot == pCur.pgnoRoot)
                    p.cachedRowid = iRowid;
            Debug.Assert(pCur.cachedRowid == iRowid);
        }

        internal static long sqlite3BtreeGetCachedRowid(BtCursor pCur) { return pCur.cachedRowid; }

        #endregion

        internal SQLITE saveCursorPosition(BtCursor pCur)
        {
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(pCur.pKey == null);
            Debug.Assert(cursorHoldsMutex(pCur));
            var rc = sqlite3BtreeKeySize(pCur, ref pCur.nKey);
            Debug.Assert(rc == SQLITE.OK);  // KeySize() cannot fail
            // If this is an intKey table, then the above call to BtreeKeySize() stores the integer key in pCur.nKey. In this case this value is
            // all that is required. Otherwise, if pCur is not open on an intKey table, then malloc space for and store the pCur.nKey bytes of key data.
            if (pCur.apPage[0].intKey == 0)
            {
                var pKey = MallocEx.sqlite3Malloc((int)pCur.nKey);
                rc = sqlite3BtreeKey(pCur, 0, (uint)pCur.nKey, pKey);
                if (rc == SQLITE.OK)
                    pCur.pKey = pKey;
            }
            Debug.Assert(pCur.apPage[0].intKey == 0 || pCur.pKey == null);
            if (rc == SQLITE.OK)
            {
                for (var i = 0; i <= pCur.iPage; i++)
                {
                    releasePage(pCur.apPage[i]);
                    pCur.apPage[i] = null;
                }
                pCur.iPage = -1;
                pCur.eState = CURSOR.REQUIRESEEK;
            }
            invalidateOverflowCache(pCur);
            return rc;
        }

        internal static void sqlite3BtreeClearCursor(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            MallocEx.sqlite3_free(ref pCur.pKey);
            pCur.eState = CURSOR.INVALID;
        }

        internal static SQLITE btreeMoveto(BtCursor pCur, byte[] pKey, long nKey, int bias, ref int pRes)
        {
            UnpackedRecord pIdxKey;   // Unpacked index key
            var aSpace = new UnpackedRecord();
            if (pKey != null)
            {
                Debug.Assert(nKey == (long)(int)nKey);
                pIdxKey = sqlite3VdbeRecordUnpack(pCur.pKeyInfo, (int)nKey, pKey, aSpace, 16);
            }
            else
                pIdxKey = null;
            var rc = sqlite3BtreeMovetoUnpacked(pCur, pIdxKey, nKey, bias != 0 ? 1 : 0, ref pRes);
            if (pKey != null)
                sqlite3VdbeDeleteUnpackedRecord(pIdxKey);
            return rc;
        }

        internal static SQLITE btreeRestoreCursorPosition(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState >= CURSOR.REQUIRESEEK);
            if (pCur.eState == CURSOR.FAULT)
                return pCur.skipNext;
            pCur.eState = CURSOR.INVALID;
            var rc = btreeMoveto(pCur, pCur.pKey, pCur.nKey, 0, ref pCur.skipNext);
            if (rc == SQLITE.OK)
            {
                pCur.pKey = null;
                Debug.Assert(pCur.eState == CURSOR.VALID || pCur.eState == CURSOR.INVALID);
            }
            return rc;
        }

        internal static SQLITE restoreCursorPosition(BtCursor pCur) { return (pCur.eState >= CURSOR.REQUIRESEEK ? btreeRestoreCursorPosition(pCur) : SQLITE.OK); }

        internal static SQLITE sqlite3BtreeCursorHasMoved(BtCursor pCur, ref int pHasMoved)
        {
            var rc = restoreCursorPosition(pCur);
            if (rc != SQLITE.OK)
            {
                pHasMoved = 1;
                return rc;
            }
            pHasMoved = (pCur.eState != CURSOR.VALID || pCur.skipNext != 0 ? 1 : 0);
            return SQLITE.OK;
        }

        internal static SQLITE sqlite3BtreeCloseCursor(BtCursor pCur)
        {
            var pBtree = pCur.pBtree;
            if (pBtree != null)
            {
                var pBt = pCur.pBt;
                sqlite3BtreeEnter(pBtree);
                sqlite3BtreeClearCursor(pCur);
                if (pCur.pPrev != null)
                    pCur.pPrev.pNext = pCur.pNext;
                else
                    pBt.pCursor = pCur.pNext;
                if (pCur.pNext != null)
                    pCur.pNext.pPrev = pCur.pPrev;
                for (var i = 0; i <= pCur.iPage; i++)
                    releasePage(pCur.apPage[i]);
                unlockBtreeIfUnused(pBt);
                invalidateOverflowCache(pCur);
                sqlite3BtreeLeave(pBtree);
            }
            return SQLITE.OK;
        }

#if !NDEBUG
        internal static void assertCellInfo(BtCursor pCur)
        {
            var iPage = pCur.iPage;
            var info = new CellInfo();
            btreeParseCell(pCur.apPage[iPage], pCur.aiIdx[iPage], ref info);
            Debug.Assert(info.GetHashCode() == pCur.info.GetHashCode() || info.Equals(pCur.info));
        }
#else
        internal static void assertCellInfo(BtCursor pCur) { }
#endif

#if true //!_MSC_VER
        internal static void getCellInfo(BtCursor pCur)
        {
            if (pCur.info.nSize == 0)
            {
                var iPage = pCur.iPage;
                btreeParseCell(pCur.apPage[iPage], pCur.aiIdx[iPage], ref pCur.info);
                pCur.validNKey = true;
            }
            else
                assertCellInfo(pCur);
        }
#else
        /* Use a macro in all other compilers so that the function is inlined */
        //#define getCellInfo(pCur)                                                      \
        //  if( pCur.info.nSize==null ){                                                   \
        //    int iPage = pCur.iPage;                                                   \
        //    btreeParseCell(pCur.apPage[iPage],pCur.aiIdx[iPage],&pCur.info); \
        //    pCur.validNKey = true;                                                       \
        //  }else{                                                                       \
        //    assertCellInfo(pCur);                                                      \
        //  }
#endif

#if DEBUG
        internal static bool sqlite3BtreeCursorIsValid(BtCursor pCur) { return pCur != null && pCur.eState == CURSOR.VALID; }
#else
        internal static bool sqlite3BtreeCursorIsValid(BtCursor pCur) { return true; }
#endif

        internal static SQLITE sqlite3BtreeKeySize(BtCursor pCur, ref long pSize)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.INVALID || pCur.eState == CURSOR.VALID);
            if (pCur.eState != CURSOR.VALID)
                pSize = 0;
            else
            {
                getCellInfo(pCur);
                pSize = pCur.info.nKey;
            }
            return SQLITE.OK;
        }

        internal static SQLITE sqlite3BtreeDataSize(BtCursor pCur, ref uint pSize)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.VALID);
            getCellInfo(pCur);
            pSize = pCur.info.nData;
            return SQLITE.OK;
        }

        internal static SQLITE copyPayload(byte[] pPayload, uint payloadOffset, byte[] pBuf, uint pBufOffset, uint nByte, int eOp, DbPage pDbPage)
        {
            if (eOp != 0)
            {
                // Copy data from buffer to page (a write operation)
                var rc = Pager.sqlite3PagerWrite(pDbPage);
                if (rc != SQLITE.OK)
                    return rc;
                Buffer.BlockCopy(pBuf, (int)pBufOffset, pPayload, (int)payloadOffset, (int)nByte);
            }
            else
                // Copy data from page to buffer (a read operation)
                Buffer.BlockCopy(pPayload, (int)payloadOffset, pBuf, (int)pBufOffset, (int)nByte);
            return SQLITE.OK;
        }

        internal static SQLITE accessPayload(BtCursor pCur, uint offset, uint amt, byte[] pBuf, int eOp)
        {
            uint pBufOffset = 0;
            byte[] aPayload;
            var rc = SQLITE.OK;
            int iIdx = 0;
            var pPage = pCur.apPage[pCur.iPage]; // Btree page of current entry
            var pBt = pCur.pBt;                  // Btree this cursor belongs to
            Debug.Assert(pPage != null);
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(pCur.aiIdx[pCur.iPage] < pPage.nCell);
            Debug.Assert(cursorHoldsMutex(pCur));
            getCellInfo(pCur);
            aPayload = pCur.info.pCell; //pCur.info.pCell + pCur.info.nHeader;
            var nKey = (uint)(pPage.intKey != 0 ? 0 : (int)pCur.info.nKey);
            if (Check.NEVER(offset + amt > nKey + pCur.info.nData) || pCur.info.nLocal > pBt.usableSize)
                // Trying to read or write past the end of the data is an error
                return SysEx.SQLITE_CORRUPT_BKPT();
            // Check if data must be read/written to/from the btree page itself.
            if (offset < pCur.info.nLocal)
            {
                var a = (int)amt;
                if (a + offset > pCur.info.nLocal)
                    a = (int)(pCur.info.nLocal - offset);
                rc = copyPayload(aPayload, (uint)(offset + pCur.info.iCell + pCur.info.nHeader), pBuf, pBufOffset, (uint)a, eOp, pPage.pDbPage);
                offset = 0;
                pBufOffset += (uint)a;
                amt -= (uint)a;
            }
            else
                offset -= pCur.info.nLocal;
            if (rc == SQLITE.OK && amt > 0)
            {
                var ovflSize = (uint)(pBt.usableSize - 4);  // Bytes content per ovfl page
                Pgno nextPage = ConvertEx.sqlite3Get4byte(aPayload, pCur.info.nLocal + pCur.info.iCell + pCur.info.nHeader);
#if !SQLITE_OMIT_INCRBLOB
/* If the isIncrblobHandle flag is set and the BtCursor.aOverflow[]
** has not been allocated, allocate it now. The array is sized at
** one entry for each overflow page in the overflow chain. The
** page number of the first overflow page is stored in aOverflow[0],
** etc. A value of 0 in the aOverflow[] array means "not yet known"
** (the cache is lazily populated).
*/
if( pCur.isIncrblobHandle && !pCur.aOverflow ){
int nOvfl = (pCur.info.nPayload-pCur.info.nLocal+ovflSize-1)/ovflSize;
pCur.aOverflow = (Pgno *)sqlite3MallocZero(sizeof(Pgno)*nOvfl);
/* nOvfl is always positive.  If it were zero, fetchPayload would have
** been used instead of this routine. */
if( ALWAYS(nOvfl) && !pCur.aOverflow ){
rc = SQLITE_NOMEM;
}
}

/* If the overflow page-list cache has been allocated and the
** entry for the first required overflow page is valid, skip
** directly to it.
*/
if( pCur.aOverflow && pCur.aOverflow[offset/ovflSize] ){
iIdx = (offset/ovflSize);
nextPage = pCur.aOverflow[iIdx];
offset = (offset%ovflSize);
}
#endif
                for (; rc == SQLITE.OK && amt > 0 && nextPage != 0; iIdx++)
                {
#if !SQLITE_OMIT_INCRBLOB
/* If required, populate the overflow page-list cache. */
if( pCur.aOverflow ){
Debug.Assert(!pCur.aOverflow[iIdx] || pCur.aOverflow[iIdx]==nextPage);
pCur.aOverflow[iIdx] = nextPage;
}
#endif
                    MemPage MemPageDummy = null;
                    if (offset >= ovflSize)
                    {
                        // The only reason to read this page is to obtain the page number for the next page in the overflow chain. The page
                        // data is not required. So first try to lookup the overflow page-list cache, if any, then fall back to the getOverflowPage() function.
#if !SQLITE_OMIT_INCRBLOB
if( pCur.aOverflow && pCur.aOverflow[iIdx+1] ){
nextPage = pCur.aOverflow[iIdx+1];
} else
#endif
                        rc = getOverflowPage(pBt, nextPage, out MemPageDummy, out nextPage);
                        offset -= ovflSize;
                    }
                    else
                    {
                        // Need to read this page properly. It contains some of the range of data that is being read (eOp==null) or written (eOp!=null).
                        var pDbPage = new PgHdr();
                        var a = (int)amt;
                        rc = pBt.pPager.sqlite3PagerGet(nextPage, ref pDbPage);
                        if (rc == SQLITE.OK)
                        {
                            aPayload = Pager.sqlite3PagerGetData(pDbPage);
                            nextPage = ConvertEx.sqlite3Get4byte(aPayload);
                            if (a + offset > ovflSize)
                                a = (int)(ovflSize - offset);
                            rc = copyPayload(aPayload, offset + 4, pBuf, pBufOffset, (uint)a, eOp, pDbPage);
                            Pager.sqlite3PagerUnref(pDbPage);
                            offset = 0;
                            amt -= (uint)a;
                            pBufOffset += (uint)a;
                        }
                    }
                }
            }
            if (rc == SQLITE.OK && amt > 0)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return rc;
        }

        internal static SQLITE sqlite3BtreeKey(BtCursor pCur, uint offset, uint amt, byte[] pBuf)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(pCur.iPage >= 0 && pCur.apPage[pCur.iPage] != null);
            Debug.Assert(pCur.aiIdx[pCur.iPage] < pCur.apPage[pCur.iPage].nCell);
            return accessPayload(pCur, offset, amt, pBuf, 0);
        }

        internal static SQLITE sqlite3BtreeData(BtCursor pCur, uint offset, uint amt, byte[] pBuf)
        {
#if !SQLITE_OMIT_INCRBLOB
            if (pCur.eState == CURSOR.INVALID)
                return SQLITE.ABORT;
#endif
            Debug.Assert(cursorHoldsMutex(pCur));
            var rc = restoreCursorPosition(pCur);
            if (rc == SQLITE.OK)
            {
                Debug.Assert(pCur.eState == CURSOR.VALID);
                Debug.Assert(pCur.iPage >= 0 && pCur.apPage[pCur.iPage] != null);
                Debug.Assert(pCur.aiIdx[pCur.iPage] < pCur.apPage[pCur.iPage].nCell);
                rc = accessPayload(pCur, offset, amt, pBuf, 0);
            }
            return rc;
        }

        internal static byte[] fetchPayload(BtCursor pCur, ref int pAmt, ref int outOffset, bool skipKey)
        {
            Debug.Assert(pCur != null && pCur.iPage >= 0 && pCur.apPage[pCur.iPage] != null);
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(cursorHoldsMutex(pCur));
            outOffset = -1;
            var pPage = pCur.apPage[pCur.iPage];
            Debug.Assert(pCur.aiIdx[pCur.iPage] < pPage.nCell);
            if (Check.NEVER(pCur.info.nSize == 0))
                btreeParseCell(pCur.apPage[pCur.iPage], pCur.aiIdx[pCur.iPage], ref pCur.info);
            var aPayload = MallocEx.sqlite3Malloc(pCur.info.nSize - pCur.info.nHeader);
            var nKey = (pPage.intKey != 0 ? 0 : (uint)pCur.info.nKey);
            uint nLocal;
            if (skipKey)
            {
                outOffset = (int)(pCur.info.iCell + pCur.info.nHeader + nKey);
                Buffer.BlockCopy(pCur.info.pCell, outOffset, aPayload, 0, (int)(pCur.info.nSize - pCur.info.nHeader - nKey));
                nLocal = pCur.info.nLocal - nKey;
            }
            else
            {
                outOffset = (int)(pCur.info.iCell + pCur.info.nHeader);
                Buffer.BlockCopy(pCur.info.pCell, outOffset, aPayload, 0, pCur.info.nSize - pCur.info.nHeader);
                nLocal = pCur.info.nLocal;
                Debug.Assert(nLocal <= nKey);
            }
            pAmt = (int)nLocal;
            return aPayload;
        }

        internal static byte[] sqlite3BtreeKeyFetch(BtCursor pCur, ref int pAmt, ref int outOffset)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pCur.pBtree.db.mutex));
            Debug.Assert(cursorHoldsMutex(pCur));
            byte[] p = null;
            if (Check.ALWAYS(pCur.eState == CURSOR.VALID))
                p = fetchPayload(pCur, ref pAmt, ref outOffset, false);
            return p;
        }

        //TODO remove sqlite3BtreeKeyFetch or sqlite3BtreeDataFetch
        internal static byte[] sqlite3BtreeDataFetch(BtCursor pCur, ref int pAmt, ref int outOffset)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pCur.pBtree.db.mutex));
            Debug.Assert(cursorHoldsMutex(pCur));
            byte[] p = null;
            if (Check.ALWAYS(pCur.eState == CURSOR.VALID))
                p = fetchPayload(pCur, ref pAmt, ref outOffset, true);
            return p;
        }

        internal static SQLITE moveToChild(BtCursor pCur, Pgno newPgno)
        {
            var i = pCur.iPage;
            var pNewPage = new MemPage();
            var pBt = pCur.pBt;
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(pCur.iPage < BTCURSOR_MAX_DEPTH);
            if (pCur.iPage >= (BTCURSOR_MAX_DEPTH - 1))
                return SysEx.SQLITE_CORRUPT_BKPT();
            var rc = getAndInitPage(pBt, newPgno, ref pNewPage);
            if (rc != SQLITE.OK)
                return rc;
            pCur.apPage[i + 1] = pNewPage;
            pCur.aiIdx[i + 1] = 0;
            pCur.iPage++;
            pCur.info.nSize = 0;
            pCur.validNKey = false;
            if (pNewPage.nCell < 1 || pNewPage.intKey != pCur.apPage[i].intKey)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return SQLITE.OK;
        }

        internal static void moveToParent(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(pCur.iPage > 0);
            Debug.Assert(pCur.apPage[pCur.iPage] != null);
            assertParentIndex(pCur.apPage[pCur.iPage - 1], pCur.aiIdx[pCur.iPage - 1], pCur.apPage[pCur.iPage].pgno);
            releasePage(pCur.apPage[pCur.iPage]);
            pCur.iPage--;
            pCur.info.nSize = 0;
            pCur.validNKey = false;
        }

        internal static SQLITE moveToRoot(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(CURSOR.INVALID < CURSOR.REQUIRESEEK);
            Debug.Assert(CURSOR.VALID < CURSOR.REQUIRESEEK);
            Debug.Assert(CURSOR.FAULT > CURSOR.REQUIRESEEK);
            if (pCur.eState >= CURSOR.REQUIRESEEK)
            {
                if (pCur.eState == CURSOR.FAULT)
                {
                    Debug.Assert(pCur.skipNext != SQLITE.OK);
                    return pCur.skipNext;
                }
                sqlite3BtreeClearCursor(pCur);
            }
            var rc = SQLITE.OK;
            var p = pCur.pBtree;
            var pBt = p.pBt;
            if (pCur.iPage >= 0)
            {
                for (var i = 1; i <= pCur.iPage; i++)
                    releasePage(pCur.apPage[i]);
                pCur.iPage = 0;
            }
            else
            {
                rc = getAndInitPage(pBt, pCur.pgnoRoot, ref pCur.apPage[0]);
                if (rc != SQLITE.OK)
                {
                    pCur.eState = CURSOR.INVALID;
                    return rc;
                }
                pCur.iPage = 0;
                // If pCur.pKeyInfo is not NULL, then the caller that opened this cursor expected to open it on an index b-tree. Otherwise, if pKeyInfo is
                // NULL, the caller expects a table b-tree. If this is not the case, return an SQLITE_CORRUPT error.
                Debug.Assert(pCur.apPage[0].intKey == 1 || pCur.apPage[0].intKey == 0);
                if ((pCur.pKeyInfo == null) != (pCur.apPage[0].intKey != 0))
                    return SysEx.SQLITE_CORRUPT_BKPT();
            }
            // Assert that the root page is of the correct type. This must be the case as the call to this function that loaded the root-page (either
            // this call or a previous invocation) would have detected corruption if the assumption were not true, and it is not possible for the flags
            // byte to have been modified while this cursor is holding a reference to the page.
            var pRoot = pCur.apPage[0];
            Debug.Assert(pRoot.pgno == pCur.pgnoRoot);
            Debug.Assert(pRoot.isInit != 0 && (pCur.pKeyInfo == null) == (pRoot.intKey != 0));
            pCur.aiIdx[0] = 0;
            pCur.info.nSize = 0;
            pCur.atLast = 0;
            pCur.validNKey = false;
            if (pRoot.nCell == 0 && 0 == pRoot.leaf)
            {
                if (pRoot.pgno != 1)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                Pgno subpage = ConvertEx.sqlite3Get4byte(pRoot.aData, pRoot.hdrOffset + 8);
                pCur.eState = CURSOR.VALID;
                rc = moveToChild(pCur, subpage);
            }
            else
                pCur.eState = (pRoot.nCell > 0 ? CURSOR.VALID : CURSOR.INVALID);
            return rc;
        }

        internal static SQLITE moveToLeftmost(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.VALID);
            MemPage pPage;
            var rc = SQLITE.OK;
            while (rc == SQLITE.OK && 0 == (pPage = pCur.apPage[pCur.iPage]).leaf)
            {
                Debug.Assert(pCur.aiIdx[pCur.iPage] < pPage.nCell);
                Pgno pgno = ConvertEx.sqlite3Get4byte(pPage.aData, findCell(pPage, pCur.aiIdx[pCur.iPage]));
                rc = moveToChild(pCur, pgno);
            }
            return rc;
        }

        internal static SQLITE moveToRightmost(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState == CURSOR.VALID);
            var rc = SQLITE.OK;
            MemPage pPage = null;
            while (rc == SQLITE.OK && 0 == (pPage = pCur.apPage[pCur.iPage]).leaf)
            {
                Pgno pgno = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                pCur.aiIdx[pCur.iPage] = pPage.nCell;
                rc = moveToChild(pCur, pgno);
            }
            if (rc == SQLITE.OK)
            {
                pCur.aiIdx[pCur.iPage] = (ushort)(pPage.nCell - 1);
                pCur.info.nSize = 0;
                pCur.validNKey = false;
            }
            return rc;
        }

        internal static SQLITE sqlite3BtreeFirst(BtCursor pCur, ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(MutexEx.sqlite3_mutex_held(pCur.pBtree.db.mutex));
            var rc = moveToRoot(pCur);
            if (rc == SQLITE.OK)
                if (pCur.eState == CURSOR.INVALID)
                {
                    Debug.Assert(pCur.apPage[pCur.iPage].nCell == 0);
                    pRes = 1;
                }
                else
                {
                    Debug.Assert(pCur.apPage[pCur.iPage].nCell > 0);
                    pRes = 0;
                    rc = moveToLeftmost(pCur);
                }
            return rc;
        }

        internal static SQLITE sqlite3BtreeLast(BtCursor pCur, ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(MutexEx.sqlite3_mutex_held(pCur.pBtree.db.mutex));
            // If the cursor already points to the last entry, this is a no-op.
            if (pCur.eState == CURSOR.VALID && pCur.atLast != 0)
            {
#if DEBUG
                // This block serves to Debug.Assert() that the cursor really does point to the last entry in the b-tree.
                for (var ii = 0; ii < pCur.iPage; ii++)
                    Debug.Assert(pCur.aiIdx[ii] == pCur.apPage[ii].nCell);
                Debug.Assert(pCur.aiIdx[pCur.iPage] == pCur.apPage[pCur.iPage].nCell - 1);
                Debug.Assert(pCur.apPage[pCur.iPage].leaf != 0);
#endif
                return SQLITE.OK;
            }
            var rc = moveToRoot(pCur);
            if (rc == SQLITE.OK)
                if (pCur.eState == CURSOR.INVALID)
                {
                    Debug.Assert(pCur.apPage[pCur.iPage].nCell == 0);
                    pRes = 1;
                }
                else
                {
                    Debug.Assert(pCur.eState == CURSOR.VALID);
                    pRes = 0;
                    rc = moveToRightmost(pCur);
                    pCur.atLast = (byte)(rc == SQLITE.OK ? 1 : 0);
                }
            return rc;
        }

        internal static SQLITE sqlite3BtreeMovetoUnpacked(BtCursor pCur, Btree.UnpackedRecord pIdxKey, long intKey, int biasRight, ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(MutexEx.sqlite3_mutex_held(pCur.pBtree.db.mutex));
            Debug.Assert((pIdxKey == null) == (pCur.pKeyInfo == null));
            // If the cursor is already positioned at the point we are trying to move to, then just return without doing any work
            if (pCur.eState == CURSOR.VALID && pCur.validNKey && pCur.apPage[0].intKey != 0)
            {
                if (pCur.info.nKey == intKey)
                {
                    pRes = 0;
                    return SQLITE.OK;
                }
                if (pCur.atLast != 0 && pCur.info.nKey < intKey)
                {
                    pRes = -1;
                    return SQLITE.OK;
                }
            }
            var rc = moveToRoot(pCur);
            if (rc != 0)
                return rc;
            Debug.Assert(pCur.apPage[pCur.iPage] != null);
            Debug.Assert(pCur.apPage[pCur.iPage].isInit != 0);
            Debug.Assert(pCur.apPage[pCur.iPage].nCell > 0 || pCur.eState == CURSOR.INVALID);
            if (pCur.eState == CURSOR.INVALID)
            {
                pRes = -1;
                Debug.Assert(pCur.apPage[pCur.iPage].nCell == 0);
                return SQLITE.OK;
            }
            Debug.Assert(pCur.apPage[0].intKey != 0 || pIdxKey != null);
            for (; ; )
            {
                Pgno chldPg;
                var pPage = pCur.apPage[pCur.iPage];
                // pPage.nCell must be greater than zero. If this is the root-page the cursor would have been INVALID above and this for(;;) loop
                // not run. If this is not the root-page, then the moveToChild() routine would have already detected db corruption. Similarly, pPage must
                // be the right kind (index or table) of b-tree page. Otherwise a moveToChild() or moveToRoot() call would have detected corruption.
                Debug.Assert(pPage.nCell > 0);
                Debug.Assert(pPage.intKey == ((pIdxKey == null) ? 1 : 0));
                var lwr = 0;
                var upr = pPage.nCell - 1;
                int idx;
                pCur.aiIdx[pCur.iPage] = (ushort)(biasRight != 0 ? (idx = upr) : (idx = (upr + lwr) / 2));
                for (; ; )
                {
                    int c;
                    Debug.Assert(idx == pCur.aiIdx[pCur.iPage]);
                    pCur.info.nSize = 0;
                    int pCell = findCell(pPage, idx) + pPage.childPtrSize; // Pointer to current cell in pPage
                    if (pPage.intKey != 0)
                    {
                        var nCellKey = 0L;
                        if (pPage.hasData != 0)
                        {
                            uint dummy0;
                            pCell += ConvertEx.getVarint32(pPage.aData, pCell, out dummy0);
                        }
                        ConvertEx.getVarint(pPage.aData, pCell, out nCellKey);
                        if (nCellKey == intKey)
                            c = 0;
                        else if (nCellKey < intKey)
                            c = -1;
                        else
                        {
                            Debug.Assert(nCellKey > intKey);
                            c = +1;
                        }
                        pCur.validNKey = true;
                        pCur.info.nKey = nCellKey;
                    }
                    else
                    {
                        // The maximum supported page-size is 65536 bytes. This means that the maximum number of record bytes stored on an index B-Tree
                        // page is less than 16384 bytes and may be stored as a 2-byte varint. This information is used to attempt to avoid parsing
                        // the entire cell by checking for the cases where the record is stored entirely within the b-tree page by inspecting the first
                        // 2 bytes of the cell.
                        var nCell = (int)pPage.aData[pCell + 0];
                        if (0 == (nCell & 0x80) && nCell <= pPage.maxLocal)
                            // This branch runs if the record-size field of the cell is a single byte varint and the record fits entirely on the main b-tree page.
                            c = sqlite3VdbeRecordCompare(nCell, pPage.aData, pCell + 1, pIdxKey);
                        else if (0 == (pPage.aData[pCell + 1] & 0x80) && (nCell = ((nCell & 0x7f) << 7) + pPage.aData[pCell + 1]) <= pPage.maxLocal)
                            // The record-size field is a 2 byte varint and the record fits entirely on the main b-tree page.
                            c = sqlite3VdbeRecordCompare(nCell, pPage.aData, pCell + 2, pIdxKey);
                        else
                        {
                            // The record flows over onto one or more overflow pages. In this case the whole cell needs to be parsed, a buffer allocated
                            // and accessPayload() used to retrieve the record into the buffer before VdbeRecordCompare() can be called.
                            var pCellBody = new byte[pPage.aData.Length - pCell + pPage.childPtrSize];
                            Buffer.BlockCopy(pPage.aData, pCell - pPage.childPtrSize, pCellBody, 0, pCellBody.Length);
                            btreeParseCellPtr(pPage, pCellBody, ref pCur.info);
                            nCell = (int)pCur.info.nKey;
                            var pCellKey = MallocEx.sqlite3Malloc(nCell);
                            rc = accessPayload(pCur, 0, (uint)nCell, pCellKey, 0);
                            if (rc != SQLITE.OK)
                            {
                                pCellKey = null;
                                goto moveto_finish;
                            }
                            c = sqlite3VdbeRecordCompare(nCell, pCellKey, pIdxKey);
                            pCellKey = null;
                        }
                    }
                    if (c == 0)
                    {
                        if (pPage.intKey != 0 && 0 == pPage.leaf)
                        {
                            lwr = idx;
                            upr = lwr - 1;
                            break;
                        }
                        else
                        {
                            pRes = 0;
                            rc = SQLITE.OK;
                            goto moveto_finish;
                        }
                    }
                    if (c < 0)
                        lwr = idx + 1;
                    else
                        upr = idx - 1;
                    if (lwr > upr)
                        break;
                    pCur.aiIdx[pCur.iPage] = (ushort)(idx = (lwr + upr) / 2);
                }
                Debug.Assert(lwr == upr + 1);
                Debug.Assert(pPage.isInit != 0);
                if (pPage.leaf != 0)
                    chldPg = 0;
                else if (lwr >= pPage.nCell)
                    chldPg = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                else
                    chldPg = ConvertEx.sqlite3Get4byte(pPage.aData, findCell(pPage, lwr));
                if (chldPg == 0)
                {
                    Debug.Assert(pCur.aiIdx[pCur.iPage] < pCur.apPage[pCur.iPage].nCell);
                    pRes = c;
                    rc = SQLITE.OK;
                    goto moveto_finish;
                }
                pCur.aiIdx[pCur.iPage] = (ushort)lwr;
                pCur.info.nSize = 0;
                pCur.validNKey = false;
                rc = moveToChild(pCur, chldPg);
                if (rc != 0)
                    goto moveto_finish;
            }
        moveto_finish:
            return rc;
        }

        internal static bool sqlite3BtreeEof(BtCursor pCur)
        {
            // TODO: What if the cursor is in CURSOR_REQUIRESEEK but all table entries have been deleted? This API will need to change to return an error code
            // as well as the boolean result value.
            return (pCur.eState != CURSOR.VALID);
        }

        internal static SQLITE sqlite3BtreeNext(BtCursor pCur, ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            var rc = restoreCursorPosition(pCur);
            if (rc != SQLITE.OK)
                return rc;
            if (pCur.eState == CURSOR.INVALID)
            {
                pRes = 1;
                return SQLITE.OK;
            }
            if (pCur.skipNext > 0)
            {
                pCur.skipNext = 0;
                pRes = 0;
                return SQLITE.OK;
            }
            pCur.skipNext = 0;
            //
            var pPage = pCur.apPage[pCur.iPage];
            var idx = ++pCur.aiIdx[pCur.iPage];
            Debug.Assert(pPage.isInit != 0);
            Debug.Assert(idx <= pPage.nCell);
            pCur.info.nSize = 0;
            pCur.validNKey = false;
            if (idx >= pPage.nCell)
            {
                if (pPage.leaf == 0)
                {
                    rc = moveToChild(pCur, ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8));
                    if (rc != SQLITE.OK)
                        return rc;
                    rc = moveToLeftmost(pCur);
                    pRes = 0;
                    return rc;
                }
                do
                {
                    if (pCur.iPage == 0)
                    {
                        pRes = 1;
                        pCur.eState = CURSOR.INVALID;
                        return SQLITE.OK;
                    }
                    moveToParent(pCur);
                    pPage = pCur.apPage[pCur.iPage];
                } while (pCur.aiIdx[pCur.iPage] >= pPage.nCell);
                pRes = 0;
                rc = (pPage.intKey != 0 ? sqlite3BtreeNext(pCur, ref pRes) : SQLITE.OK);
                return rc;
            }
            pRes = 0;
            if (pPage.leaf != 0)
                return SQLITE.OK;
            rc = moveToLeftmost(pCur);
            return rc;
        }

        internal static SQLITE sqlite3BtreePrevious(BtCursor pCur, ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            var rc = restoreCursorPosition(pCur);
            if (rc != SQLITE.OK)
                return rc;
            pCur.atLast = 0;
            if (pCur.eState == CURSOR.INVALID)
            {
                pRes = 1;
                return SQLITE.OK;
            }
            if (pCur.skipNext < 0)
            {
                pCur.skipNext = 0;
                pRes = 0;
                return SQLITE.OK;
            }
            pCur.skipNext = 0;
            //
            var pPage = pCur.apPage[pCur.iPage];
            Debug.Assert(pPage.isInit != 0);
            if (pPage.leaf == 0)
            {
                var idx = pCur.aiIdx[pCur.iPage];
                rc = moveToChild(pCur, ConvertEx.sqlite3Get4byte(pPage.aData, findCell(pPage, idx)));
                if (rc != SQLITE.OK)
                    return rc;
                rc = moveToRightmost(pCur);
            }
            else
            {
                while (pCur.aiIdx[pCur.iPage] == 0)
                {
                    if (pCur.iPage == 0)
                    {
                        pCur.eState = CURSOR.INVALID;
                        pRes = 1;
                        return SQLITE.OK;
                    }
                    moveToParent(pCur);
                }
                pCur.info.nSize = 0;
                pCur.validNKey = false;
                pCur.aiIdx[pCur.iPage]--;
                pPage = pCur.apPage[pCur.iPage];
                rc = (pPage.intKey != 0 && 0 == pPage.leaf ? sqlite3BtreePrevious(pCur, ref pRes) : SQLITE.OK);
            }
            pRes = 0;
            return rc;
        }

#if !SQLITE_OMIT_BTREECOUNT
        internal static SQLITE sqlite3BtreeCount(BtCursor pCur, ref long pnEntry)
        {
            long nEntry = 0; // Value to return in pnEntry
            var rc = moveToRoot(pCur);
            // Unless an error occurs, the following loop runs one iteration for each page in the B-Tree structure (not including overflow pages).
            while (rc == SQLITE.OK)
            {
                // If this is a leaf page or the tree is not an int-key tree, then this page contains countable entries. Increment the entry counter accordingly.
                var pPage = pCur.apPage[pCur.iPage]; // Current page of the b-tree 
                if (pPage.leaf != 0 || 0 == pPage.intKey)
                    nEntry += pPage.nCell;
                // pPage is a leaf node. This loop navigates the cursor so that it points to the first interior cell that it points to the parent of
                // the next page in the tree that has not yet been visited. The pCur.aiIdx[pCur.iPage] value is set to the index of the parent cell
                // of the page, or to the number of cells in the page if the next page to visit is the right-child of its parent.
                // If all pages in the tree have been visited, return SQLITE_OK to the caller.
                if (pPage.leaf != 0)
                {
                    do
                    {
                        if (pCur.iPage == 0)
                        {
                            // All pages of the b-tree have been visited. Return successfully.
                            pnEntry = nEntry;
                            return SQLITE.OK;
                        }
                        moveToParent(pCur);
                    } while (pCur.aiIdx[pCur.iPage] >= pCur.apPage[pCur.iPage].nCell);
                    pCur.aiIdx[pCur.iPage]++;
                    pPage = pCur.apPage[pCur.iPage];
                }
                // Descend to the child node of the cell that the cursor currently points at. This is the right-child if (iIdx==pPage.nCell).
                var iIdx = pCur.aiIdx[pCur.iPage]; // Index of child node in parent
                rc = (iIdx == pPage.nCell ? moveToChild(pCur, ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8)) : moveToChild(pCur, ConvertEx.sqlite3Get4byte(pPage.aData, findCell(pPage, iIdx))));
            }
            // An error has occurred. Return an error code.
            return rc;
        }
#endif

    }
}
