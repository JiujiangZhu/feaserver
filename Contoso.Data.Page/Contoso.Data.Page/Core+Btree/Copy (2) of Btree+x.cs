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
    public partial class Btree
    {

        internal static SQLITE sqlite3BtreeBeginStmt(Btree p, int iStatement)
        {
            sqlite3BtreeEnter(p);
            Debug.Assert(p.inTrans == TRANS.WRITE);
            var pBt = p.pBt;
            Debug.Assert(!pBt.readOnly);
            Debug.Assert(iStatement > 0);
            Debug.Assert(iStatement > p.db.nSavepoint);
            Debug.Assert(pBt.inTransaction == TRANS.WRITE);
            // At the pager level, a statement transaction is a savepoint with an index greater than all savepoints created explicitly using
            // SQL statements. It is illegal to open, release or rollback any such savepoints while the statement transaction savepoint is active.
            var rc = pBt.pPager.sqlite3PagerOpenSavepoint(iStatement);
            sqlite3BtreeLeave(p);
            return rc;
        }

        internal static SQLITE sqlite3BtreeSavepoint(Btree p, SAVEPOINT op, int iSavepoint)
        {
            var rc = SQLITE.OK;
            if (p != null && p.inTrans == TRANS.WRITE)
            {
                Debug.Assert(op == SAVEPOINT.RELEASE || op == SAVEPOINT.ROLLBACK);
                Debug.Assert(iSavepoint >= 0 || (iSavepoint == -1 && op == SAVEPOINT.ROLLBACK));
                sqlite3BtreeEnter(p);
                var pBt = p.pBt;
                rc = pBt.pPager.sqlite3PagerSavepoint(op, iSavepoint);
                if (rc == SQLITE.OK)
                {
                    if (iSavepoint < 0 && pBt.initiallyEmpty)
                        pBt.nPage = 0;
                    rc = newDatabase(pBt);
                    pBt.nPage = ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 28);
                    //The database size was written into the offset 28 of the header when the transaction started, so we know that the value at offset 28 is nonzero.
                    Debug.Assert(pBt.nPage > 0);
                }
                sqlite3BtreeLeave(p);
            }
            return rc;
        }

        internal static SQLITE btreeCursor(Btree p, int iTable, int wrFlag, KeyInfo pKeyInfo, BtCursor pCur)
        {
            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            Debug.Assert(wrFlag == 0 || wrFlag == 1);
            // The following Debug.Assert statements verify that if this is a sharable b-tree database, the connection is holding the required table locks,
            // and that no other connection has any open cursor that conflicts with this lock.
            Debug.Assert(hasSharedCacheTableLock(p, (uint)iTable, pKeyInfo != null ? 1 : 0, wrFlag + 1));
            Debug.Assert(wrFlag == 0 || !hasReadConflicts(p, (uint)iTable));
            // Assert that the caller has opened the required transaction.
            Debug.Assert(p.inTrans > TRANS.NONE);
            Debug.Assert(wrFlag == 0 || p.inTrans == TRANS.WRITE);
            var pBt = p.pBt;                 // Shared b-tree handle
            Debug.Assert(pBt.pPage1 != null && pBt.pPage1.aData != null);
            if (Check.NEVER(wrFlag != 0 && pBt.readOnly))
                return SQLITE.READONLY;
            if (iTable == 1 && btreePagecount(pBt) == 0)
                return SQLITE.EMPTY;
            // Now that no other errors can occur, finish filling in the BtCursor variables and link the cursor into the BtShared list.
            pCur.pgnoRoot = (Pgno)iTable;
            pCur.iPage = -1;
            pCur.pKeyInfo = pKeyInfo;
            pCur.pBtree = p;
            pCur.pBt = pBt;
            pCur.wrFlag = (byte)wrFlag;
            pCur.pNext = pBt.pCursor;
            if (pCur.pNext != null)
                pCur.pNext.pPrev = pCur;
            pBt.pCursor = pCur;
            pCur.eState = CURSOR.INVALID;
            pCur.cachedRowid = 0;
            return SQLITE.OK;
        }

        internal static SQLITE sqlite3BtreeCursor(Btree p, int iTable, int wrFlag, KeyInfo pKeyInfo, BtCursor pCur)
        {
            sqlite3BtreeEnter(p);
            var rc = btreeCursor(p, iTable, wrFlag, pKeyInfo, pCur);
            sqlite3BtreeLeave(p);
            return rc;
        }

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

        internal static int sqlite3BtreeCloseCursor(BtCursor pCur)
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

        internal static SQLITE getOverflowPage(BtShared pBt, Pgno ovfl, out MemPage ppPage, out Pgno pPgnoNext)
        {
            Pgno next = 0;
            MemPage pPage = null;
            ppPage = null;
            var rc = SQLITE.OK;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            // Debug.Assert( pPgnoNext != 0);
#if !SQLITE_OMIT_AUTOVACUUM
            // Try to find the next page in the overflow list using the autovacuum pointer-map pages. Guess that the next page in
            // the overflow list is page number (ovfl+1). If that guess turns out to be wrong, fall back to loading the data of page
            // number ovfl to determine the next page number.
            if (pBt.autoVacuum)
            {
                Pgno pgno = 0;
                Pgno iGuess = ovfl + 1;
                byte eType = 0;
                while (PTRMAP_ISPAGE(pBt, iGuess) || iGuess == PENDING_BYTE_PAGE(pBt))
                    iGuess++;
                if (iGuess <= btreePagecount(pBt))
                {
                    rc = ptrmapGet(pBt, iGuess, ref eType, ref pgno);
                    if (rc == SQLITE.OK && eType == PTRMAP_OVERFLOW2 && pgno == ovfl)
                    {
                        next = iGuess;
                        rc = SQLITE.DONE;
                    }
                }
            }
#endif
            Debug.Assert(next == 0 || rc == SQLITE.DONE);
            if (rc == SQLITE.OK)
            {
                rc = btreeGetPage(pBt, ovfl, ref pPage, 0);
                Debug.Assert(rc == SQLITE.OK || pPage == null);
                if (rc == SQLITE.OK)
                    next = ConvertEx.sqlite3Get4byte(pPage.aData);
            }
            pPgnoNext = next;
            if (ppPage != null)
                ppPage = pPage;
            else
                releasePage(pPage);
            return (rc == SQLITE.DONE ? SQLITE.OK : rc);
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

#if DEBUG
        static void assertParentIndex(MemPage pParent, int iIdx, Pgno iChild)
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

        static int sqlite3BtreeMovetoUnpacked(BtCursor pCur, UnpackedRecord pIdxKey, long intKey, int biasRight, ref int pRes)
        {
            int rc;

            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(sqlite3_mutex_held(pCur.pBtree.db.mutex));
            // Not needed in C# // Debug.Assert( pRes != 0 );
            Debug.Assert((pIdxKey == null) == (pCur.pKeyInfo == null));

            /* If the cursor is already positioned at the point we are trying
            ** to move to, then just return without doing any work */
            if (pCur.eState == CURSOR_VALID && pCur.validNKey
            && pCur.apPage[0].intKey != 0
            )
            {
                if (pCur.info.nKey == intKey)
                {
                    pRes = 0;
                    return SQLITE_OK;
                }
                if (pCur.atLast != 0 && pCur.info.nKey < intKey)
                {
                    pRes = -1;
                    return SQLITE_OK;
                }
            }

            rc = moveToRoot(pCur);
            if (rc != 0)
            {
                return rc;
            }
            Debug.Assert(pCur.apPage[pCur.iPage] != null);
            Debug.Assert(pCur.apPage[pCur.iPage].isInit != 0);
            Debug.Assert(pCur.apPage[pCur.iPage].nCell > 0 || pCur.eState == CURSOR_INVALID);
            if (pCur.eState == CURSOR_INVALID)
            {
                pRes = -1;
                Debug.Assert(pCur.apPage[pCur.iPage].nCell == 0);
                return SQLITE_OK;
            }
            Debug.Assert(pCur.apPage[0].intKey != 0 || pIdxKey != null);
            for (; ; )
            {
                int lwr, upr, idx;
                Pgno chldPg;
                MemPage pPage = pCur.apPage[pCur.iPage];
                int c;

                /* pPage.nCell must be greater than zero. If this is the root-page
                ** the cursor would have been INVALID above and this for(;;) loop
                ** not run. If this is not the root-page, then the moveToChild() routine
                ** would have already detected db corruption. Similarly, pPage must
                ** be the right kind (index or table) of b-tree page. Otherwise
                ** a moveToChild() or moveToRoot() call would have detected corruption.  */
                Debug.Assert(pPage.nCell > 0);
                Debug.Assert(pPage.intKey == ((pIdxKey == null) ? 1 : 0));
                lwr = 0;
                upr = pPage.nCell - 1;
                if (biasRight != 0)
                {
                    pCur.aiIdx[pCur.iPage] = (u16)(idx = upr);
                }
                else
                {
                    pCur.aiIdx[pCur.iPage] = (u16)(idx = (upr + lwr) / 2);
                }
                for (; ; )
                {
                    int pCell;                        /* Pointer to current cell in pPage */

                    Debug.Assert(idx == pCur.aiIdx[pCur.iPage]);
                    pCur.info.nSize = 0;
                    pCell = findCell(pPage, idx) + pPage.childPtrSize;
                    if (pPage.intKey != 0)
                    {
                        i64 nCellKey = 0;
                        if (pPage.hasData != 0)
                        {
                            u32 Dummy0 = 0;
                            pCell += getVarint32(pPage.aData, pCell, out Dummy0);
                        }
                        getVarint(pPage.aData, pCell, out nCellKey);
                        if (nCellKey == intKey)
                        {
                            c = 0;
                        }
                        else if (nCellKey < intKey)
                        {
                            c = -1;
                        }
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
                        /* The maximum supported page-size is 65536 bytes. This means that
                        ** the maximum number of record bytes stored on an index B-Tree
                        ** page is less than 16384 bytes and may be stored as a 2-byte
                        ** varint. This information is used to attempt to avoid parsing
                        ** the entire cell by checking for the cases where the record is
                        ** stored entirely within the b-tree page by inspecting the first
                        ** 2 bytes of the cell.
                        */
                        int nCell = pPage.aData[pCell + 0]; //pCell[0];
                        if (0 == (nCell & 0x80) && nCell <= pPage.maxLocal)
                        {
                            /* This branch runs if the record-size field of the cell is a
                            ** single byte varint and the record fits entirely on the main
                            ** b-tree page.  */
                            c = sqlite3VdbeRecordCompare(nCell, pPage.aData, pCell + 1, pIdxKey); //c = sqlite3VdbeRecordCompare( nCell, (void*)&pCell[1], pIdxKey );
                        }
                        else if (0 == (pPage.aData[pCell + 1] & 0x80)//!(pCell[1] & 0x80)
                        && (nCell = ((nCell & 0x7f) << 7) + pPage.aData[pCell + 1]) <= pPage.maxLocal//pCell[1])<=pPage.maxLocal
                        )
                        {
                            /* The record-size field is a 2 byte varint and the record
                            ** fits entirely on the main b-tree page.  */
                            c = sqlite3VdbeRecordCompare(nCell, pPage.aData, pCell + 2, pIdxKey); //c = sqlite3VdbeRecordCompare( nCell, (void*)&pCell[2], pIdxKey );
                        }
                        else
                        {
                            /* The record flows over onto one or more overflow pages. In
                            ** this case the whole cell needs to be parsed, a buffer allocated
                            ** and accessPayload() used to retrieve the record into the
                            ** buffer before VdbeRecordCompare() can be called. */
                            u8[] pCellKey;
                            u8[] pCellBody = new u8[pPage.aData.Length - pCell + pPage.childPtrSize];
                            Buffer.BlockCopy(pPage.aData, pCell - pPage.childPtrSize, pCellBody, 0, pCellBody.Length);//          u8 * const pCellBody = pCell - pPage->childPtrSize;
                            btreeParseCellPtr(pPage, pCellBody, ref pCur.info);
                            nCell = (int)pCur.info.nKey;
                            pCellKey = sqlite3Malloc(nCell);
                            //if ( pCellKey == null )
                            //{
                            //  rc = SQLITE_NOMEM;
                            //  goto moveto_finish;
                            //}
                            rc = accessPayload(pCur, 0, (u32)nCell, pCellKey, 0);
                            if (rc != 0)
                            {
                                pCellKey = null;// sqlite3_free(ref pCellKey );
                                goto moveto_finish;
                            }
                            c = sqlite3VdbeRecordCompare(nCell, pCellKey, pIdxKey);
                            pCellKey = null;// sqlite3_free(ref pCellKey );
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
                            rc = SQLITE_OK;
                            goto moveto_finish;
                        }
                    }
                    if (c < 0)
                    {
                        lwr = idx + 1;
                    }
                    else
                    {
                        upr = idx - 1;
                    }
                    if (lwr > upr)
                    {
                        break;
                    }
                    pCur.aiIdx[pCur.iPage] = (u16)(idx = (lwr + upr) / 2);
                }
                Debug.Assert(lwr == upr + 1);
                Debug.Assert(pPage.isInit != 0);
                if (pPage.leaf != 0)
                {
                    chldPg = 0;
                }
                else if (lwr >= pPage.nCell)
                {
                    chldPg = sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                }
                else
                {
                    chldPg = sqlite3Get4byte(pPage.aData, findCell(pPage, lwr));
                }
                if (chldPg == 0)
                {
                    Debug.Assert(pCur.aiIdx[pCur.iPage] < pCur.apPage[pCur.iPage].nCell);
                    pRes = c;
                    rc = SQLITE_OK;
                    goto moveto_finish;
                }
                pCur.aiIdx[pCur.iPage] = (u16)lwr;
                pCur.info.nSize = 0;
                pCur.validNKey = false;
                rc = moveToChild(pCur, chldPg);
                if (rc != 0)
                    goto moveto_finish;
            }
        moveto_finish:
            return rc;
        }


        /*
        ** Return TRUE if the cursor is not pointing at an entry of the table.
        **
        ** TRUE will be returned after a call to sqlite3BtreeNext() moves
        ** past the last entry in the table or sqlite3BtreePrev() moves past
        ** the first entry.  TRUE is also returned if the table is empty.
        */
        static bool sqlite3BtreeEof(BtCursor pCur)
        {
            /* TODO: What if the cursor is in CURSOR_REQUIRESEEK but all table entries
            ** have been deleted? This API will need to change to return an error code
            ** as well as the boolean result value.
            */
            return (CURSOR_VALID != pCur.eState);
        }

        /*
        ** Advance the cursor to the next entry in the database.  If
        ** successful then set pRes=0.  If the cursor
        ** was already pointing to the last entry in the database before
        ** this routine was called, then set pRes=1.
        */
        static int sqlite3BtreeNext(BtCursor pCur, ref int pRes)
        {
            int rc;
            int idx;
            MemPage pPage;

            Debug.Assert(cursorHoldsMutex(pCur));
            rc = restoreCursorPosition(pCur);
            if (rc != SQLITE_OK)
            {
                return rc;
            }
            // Not needed in C# // Debug.Assert( pRes != 0 );
            if (CURSOR_INVALID == pCur.eState)
            {
                pRes = 1;
                return SQLITE_OK;
            }
            if (pCur.skipNext > 0)
            {
                pCur.skipNext = 0;
                pRes = 0;
                return SQLITE_OK;
            }
            pCur.skipNext = 0;

            pPage = pCur.apPage[pCur.iPage];
            idx = ++pCur.aiIdx[pCur.iPage];
            Debug.Assert(pPage.isInit != 0);
            Debug.Assert(idx <= pPage.nCell);

            pCur.info.nSize = 0;
            pCur.validNKey = false;
            if (idx >= pPage.nCell)
            {
                if (0 == pPage.leaf)
                {
                    rc = moveToChild(pCur, sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8));
                    if (rc != 0)
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
                        pCur.eState = CURSOR_INVALID;
                        return SQLITE_OK;
                    }
                    moveToParent(pCur);
                    pPage = pCur.apPage[pCur.iPage];
                } while (pCur.aiIdx[pCur.iPage] >= pPage.nCell);
                pRes = 0;
                if (pPage.intKey != 0)
                {
                    rc = sqlite3BtreeNext(pCur, ref pRes);
                }
                else
                {
                    rc = SQLITE_OK;
                }
                return rc;
            }
            pRes = 0;
            if (pPage.leaf != 0)
            {
                return SQLITE_OK;
            }
            rc = moveToLeftmost(pCur);
            return rc;
        }


        /*
        ** Step the cursor to the back to the previous entry in the database.  If
        ** successful then set pRes=0.  If the cursor
        ** was already pointing to the first entry in the database before
        ** this routine was called, then set pRes=1.
        */
        static int sqlite3BtreePrevious(BtCursor pCur, ref int pRes)
        {
            int rc;
            MemPage pPage;

            Debug.Assert(cursorHoldsMutex(pCur));
            rc = restoreCursorPosition(pCur);
            if (rc != SQLITE_OK)
            {
                return rc;
            }
            pCur.atLast = 0;
            if (CURSOR_INVALID == pCur.eState)
            {
                pRes = 1;
                return SQLITE_OK;
            }
            if (pCur.skipNext < 0)
            {
                pCur.skipNext = 0;
                pRes = 0;
                return SQLITE_OK;
            }
            pCur.skipNext = 0;

            pPage = pCur.apPage[pCur.iPage];
            Debug.Assert(pPage.isInit != 0);
            if (0 == pPage.leaf)
            {
                int idx = pCur.aiIdx[pCur.iPage];
                rc = moveToChild(pCur, sqlite3Get4byte(pPage.aData, findCell(pPage, idx)));
                if (rc != 0)
                {
                    return rc;
                }
                rc = moveToRightmost(pCur);
            }
            else
            {
                while (pCur.aiIdx[pCur.iPage] == 0)
                {
                    if (pCur.iPage == 0)
                    {
                        pCur.eState = CURSOR_INVALID;
                        pRes = 1;
                        return SQLITE_OK;
                    }
                    moveToParent(pCur);
                }
                pCur.info.nSize = 0;
                pCur.validNKey = false;

                pCur.aiIdx[pCur.iPage]--;
                pPage = pCur.apPage[pCur.iPage];
                if (pPage.intKey != 0 && 0 == pPage.leaf)
                {
                    rc = sqlite3BtreePrevious(pCur, ref pRes);
                }
                else
                {
                    rc = SQLITE_OK;
                }
            }
            pRes = 0;
            return rc;
        }

        /*
        ** Allocate a new page from the database file.
        **
        ** The new page is marked as dirty.  (In other words, sqlite3PagerWrite()
        ** has already been called on the new page.)  The new page has also
        ** been referenced and the calling routine is responsible for calling
        ** sqlite3PagerUnref() on the new page when it is done.
        **
        ** SQLITE_OK is returned on success.  Any other return value indicates
        ** an error.  ppPage and pPgno are undefined in the event of an error.
        ** Do not invoke sqlite3PagerUnref() on ppPage if an error is returned.
        **
        ** If the "nearby" parameter is not 0, then a (feeble) effort is made to
        ** locate a page close to the page number "nearby".  This can be used in an
        ** attempt to keep related pages close to each other in the database file,
        ** which in turn can make database access faster.
        **
        ** If the "exact" parameter is not 0, and the page-number nearby exists
        ** anywhere on the free-list, then it is guarenteed to be returned. This
        ** is only used by auto-vacuum databases when allocating a new table.
        */
        static int allocateBtreePage(
        BtShared pBt,
        ref MemPage ppPage,
        ref Pgno pPgno,
        Pgno nearby,
        u8 exact
        )
        {
            MemPage pPage1;
            int rc;
            u32 n;     /* Number of pages on the freelist */
            u32 k;     /* Number of leaves on the trunk of the freelist */
            MemPage pTrunk = null;
            MemPage pPrevTrunk = null;
            Pgno mxPage;     /* Total size of the database file */

            Debug.Assert(sqlite3_mutex_held(pBt.mutex));
            pPage1 = pBt.pPage1;
            mxPage = btreePagecount(pBt);
            n = sqlite3Get4byte(pPage1.aData, 36);
            testcase(n == mxPage - 1);
            if (n >= mxPage)
            {
                return SQLITE_CORRUPT_BKPT();
            }
            if (n > 0)
            {
                /* There are pages on the freelist.  Reuse one of those pages. */
                Pgno iTrunk;
                u8 searchList = 0; /* If the free-list must be searched for 'nearby' */

                /* If the 'exact' parameter was true and a query of the pointer-map
                ** shows that the page 'nearby' is somewhere on the free-list, then
                ** the entire-list will be searched for that page.
                */
#if !SQLITE_OMIT_AUTOVACUUM
                if (exact != 0 && nearby <= mxPage)
                {
                    u8 eType = 0;
                    Debug.Assert(nearby > 0);
                    Debug.Assert(pBt.autoVacuum);
                    u32 Dummy0 = 0;
                    rc = ptrmapGet(pBt, nearby, ref eType, ref Dummy0);
                    if (rc != 0)
                        return rc;
                    if (eType == PTRMAP_FREEPAGE)
                    {
                        searchList = 1;
                    }
                    pPgno = nearby;
                }
#endif

                /* Decrement the free-list count by 1. Set iTrunk to the index of the
** first free-list trunk page. iPrevTrunk is initially 1.
*/
                rc = sqlite3PagerWrite(pPage1.pDbPage);
                if (rc != 0)
                    return rc;
                sqlite3Put4byte(pPage1.aData, (u32)36, n - 1);

                /* The code within this loop is run only once if the 'searchList' variable
                ** is not true. Otherwise, it runs once for each trunk-page on the
                ** free-list until the page 'nearby' is located.
                */
                do
                {
                    pPrevTrunk = pTrunk;
                    if (pPrevTrunk != null)
                    {
                        iTrunk = sqlite3Get4byte(pPrevTrunk.aData, 0);
                    }
                    else
                    {
                        iTrunk = sqlite3Get4byte(pPage1.aData, 32);
                    }
                    testcase(iTrunk == mxPage);
                    if (iTrunk > mxPage)
                    {
                        rc = SQLITE_CORRUPT_BKPT();
                    }
                    else
                    {
                        rc = btreeGetPage(pBt, iTrunk, ref pTrunk, 0);
                    }
                    if (rc != 0)
                    {
                        pTrunk = null;
                        goto end_allocate_page;
                    }

                    k = sqlite3Get4byte(pTrunk.aData, 4); /* # of leaves on this trunk page */
                    if (k == 0 && 0 == searchList)
                    {
                        /* The trunk has no leaves and the list is not being searched.
                        ** So extract the trunk page itself and use it as the newly
                        ** allocated page */
                        Debug.Assert(pPrevTrunk == null);
                        rc = sqlite3PagerWrite(pTrunk.pDbPage);
                        if (rc != 0)
                        {
                            goto end_allocate_page;
                        }
                        pPgno = iTrunk;
                        Buffer.BlockCopy(pTrunk.aData, 0, pPage1.aData, 32, 4);//memcpy( pPage1.aData[32], ref pTrunk.aData[0], 4 );
                        ppPage = pTrunk;
                        pTrunk = null;
                        TRACE("ALLOCATE: %d trunk - %d free pages left\n", pPgno, n - 1);
                    }
                    else if (k > (u32)(pBt.usableSize / 4 - 2))
                    {
                        /* Value of k is out of range.  Database corruption */
                        rc = SQLITE_CORRUPT_BKPT();
                        goto end_allocate_page;
#if !SQLITE_OMIT_AUTOVACUUM
                    }
                    else if (searchList != 0 && nearby == iTrunk)
                    {
                        /* The list is being searched and this trunk page is the page
                        ** to allocate, regardless of whether it has leaves.
                        */
                        Debug.Assert(pPgno == iTrunk);
                        ppPage = pTrunk;
                        searchList = 0;
                        rc = sqlite3PagerWrite(pTrunk.pDbPage);
                        if (rc != 0)
                        {
                            goto end_allocate_page;
                        }
                        if (k == 0)
                        {
                            if (null == pPrevTrunk)
                            {
                                //memcpy(pPage1.aData[32], pTrunk.aData[0], 4);
                                pPage1.aData[32 + 0] = pTrunk.aData[0 + 0];
                                pPage1.aData[32 + 1] = pTrunk.aData[0 + 1];
                                pPage1.aData[32 + 2] = pTrunk.aData[0 + 2];
                                pPage1.aData[32 + 3] = pTrunk.aData[0 + 3];
                            }
                            else
                            {
                                rc = sqlite3PagerWrite(pPrevTrunk.pDbPage);
                                if (rc != SQLITE_OK)
                                {
                                    goto end_allocate_page;
                                }
                                //memcpy(pPrevTrunk.aData[0], pTrunk.aData[0], 4);
                                pPrevTrunk.aData[0 + 0] = pTrunk.aData[0 + 0];
                                pPrevTrunk.aData[0 + 1] = pTrunk.aData[0 + 1];
                                pPrevTrunk.aData[0 + 2] = pTrunk.aData[0 + 2];
                                pPrevTrunk.aData[0 + 3] = pTrunk.aData[0 + 3];
                            }
                        }
                        else
                        {
                            /* The trunk page is required by the caller but it contains
                            ** pointers to free-list leaves. The first leaf becomes a trunk
                            ** page in this case.
                            */
                            MemPage pNewTrunk = new MemPage();
                            Pgno iNewTrunk = sqlite3Get4byte(pTrunk.aData, 8);
                            if (iNewTrunk > mxPage)
                            {
                                rc = SQLITE_CORRUPT_BKPT();
                                goto end_allocate_page;
                            }
                            testcase(iNewTrunk == mxPage);
                            rc = btreeGetPage(pBt, iNewTrunk, ref pNewTrunk, 0);
                            if (rc != SQLITE_OK)
                            {
                                goto end_allocate_page;
                            }
                            rc = sqlite3PagerWrite(pNewTrunk.pDbPage);
                            if (rc != SQLITE_OK)
                            {
                                releasePage(pNewTrunk);
                                goto end_allocate_page;
                            }
                            //memcpy(pNewTrunk.aData[0], pTrunk.aData[0], 4);
                            pNewTrunk.aData[0 + 0] = pTrunk.aData[0 + 0];
                            pNewTrunk.aData[0 + 1] = pTrunk.aData[0 + 1];
                            pNewTrunk.aData[0 + 2] = pTrunk.aData[0 + 2];
                            pNewTrunk.aData[0 + 3] = pTrunk.aData[0 + 3];
                            sqlite3Put4byte(pNewTrunk.aData, (u32)4, (u32)(k - 1));
                            Buffer.BlockCopy(pTrunk.aData, 12, pNewTrunk.aData, 8, (int)(k - 1) * 4);//memcpy( pNewTrunk.aData[8], ref pTrunk.aData[12], ( k - 1 ) * 4 );
                            releasePage(pNewTrunk);
                            if (null == pPrevTrunk)
                            {
                                Debug.Assert(sqlite3PagerIswriteable(pPage1.pDbPage));
                                sqlite3Put4byte(pPage1.aData, (u32)32, iNewTrunk);
                            }
                            else
                            {
                                rc = sqlite3PagerWrite(pPrevTrunk.pDbPage);
                                if (rc != 0)
                                {
                                    goto end_allocate_page;
                                }
                                sqlite3Put4byte(pPrevTrunk.aData, (u32)0, iNewTrunk);
                            }
                        }
                        pTrunk = null;
                        TRACE("ALLOCATE: %d trunk - %d free pages left\n", pPgno, n - 1);
#endif
                    }
                    else if (k > 0)
                    {
                        /* Extract a leaf from the trunk */
                        u32 closest;
                        Pgno iPage;
                        byte[] aData = pTrunk.aData;
                        if (nearby > 0)
                        {
                            u32 i;
                            int dist;
                            closest = 0;
                            dist = sqlite3AbsInt32((int)(sqlite3Get4byte(aData, 8) - nearby));
                            for (i = 1; i < k; i++)
                            {
                                int d2 = sqlite3AbsInt32((int)(sqlite3Get4byte(aData, 8 + i * 4) - nearby));
                                if (d2 < dist)
                                {
                                    closest = i;
                                    dist = d2;
                                }
                            }
                        }
                        else
                        {
                            closest = 0;
                        }

                        iPage = sqlite3Get4byte(aData, 8 + closest * 4);
                        testcase(iPage == mxPage);
                        if (iPage > mxPage)
                        {
                            rc = SQLITE_CORRUPT_BKPT();
                            goto end_allocate_page;
                        }
                        testcase(iPage == mxPage);
                        if (0 == searchList || iPage == nearby)
                        {
                            int noContent;
                            pPgno = iPage;
                            TRACE("ALLOCATE: %d was leaf %d of %d on trunk %d" +
                            ": %d more free pages\n",
                            pPgno, closest + 1, k, pTrunk.pgno, n - 1);
                            rc = sqlite3PagerWrite(pTrunk.pDbPage);
                            if (rc != 0)
                                goto end_allocate_page;
                            if (closest < k - 1)
                            {
                                Buffer.BlockCopy(aData, (int)(4 + k * 4), aData, 8 + (int)closest * 4, 4);//memcpy( aData[8 + closest * 4], ref aData[4 + k * 4], 4 );
                            }
                            sqlite3Put4byte(aData, (u32)4, (k - 1));// sqlite3Put4byte( aData, 4, k - 1 );
                            noContent = !btreeGetHasContent(pBt, pPgno) ? 1 : 0;
                            rc = btreeGetPage(pBt, pPgno, ref ppPage, noContent);
                            if (rc == SQLITE_OK)
                            {
                                rc = sqlite3PagerWrite((ppPage).pDbPage);
                                if (rc != SQLITE_OK)
                                {
                                    releasePage(ppPage);
                                }
                            }
                            searchList = 0;
                        }
                    }
                    releasePage(pPrevTrunk);
                    pPrevTrunk = null;
                } while (searchList != 0);
            }
            else
            {
                /* There are no pages on the freelist, so create a new page at the
                ** end of the file */
                rc = sqlite3PagerWrite(pBt.pPage1.pDbPage);
                if (rc != 0)
                    return rc;
                pBt.nPage++;
                if (pBt.nPage == PENDING_BYTE_PAGE(pBt))
                    pBt.nPage++;

#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum && PTRMAP_ISPAGE(pBt, pBt.nPage))
                {
                    /* If pPgno refers to a pointer-map page, allocate two new pages
                    ** at the end of the file instead of one. The first allocated page
                    ** becomes a new pointer-map page, the second is used by the caller.
                    */
                    MemPage pPg = null;
                    TRACE("ALLOCATE: %d from end of file (pointer-map page)\n", pPgno);
                    Debug.Assert(pBt.nPage != PENDING_BYTE_PAGE(pBt));
                    rc = btreeGetPage(pBt, pBt.nPage, ref pPg, 1);
                    if (rc == SQLITE_OK)
                    {
                        rc = sqlite3PagerWrite(pPg.pDbPage);
                        releasePage(pPg);
                    }
                    if (rc != 0)
                        return rc;
                    pBt.nPage++;
                    if (pBt.nPage == PENDING_BYTE_PAGE(pBt))
                    {
                        pBt.nPage++;
                    }
                }
#endif
                sqlite3Put4byte(pBt.pPage1.aData, (u32)28, pBt.nPage);
                pPgno = pBt.nPage;

                Debug.Assert(pPgno != PENDING_BYTE_PAGE(pBt));
                rc = btreeGetPage(pBt, pPgno, ref ppPage, 1);
                if (rc != 0)
                    return rc;
                rc = sqlite3PagerWrite((ppPage).pDbPage);
                if (rc != SQLITE_OK)
                {
                    releasePage(ppPage);
                }
                TRACE("ALLOCATE: %d from end of file\n", pPgno);
            }

            Debug.Assert(pPgno != PENDING_BYTE_PAGE(pBt));

        end_allocate_page:
            releasePage(pTrunk);
            releasePage(pPrevTrunk);
            if (rc == SQLITE_OK)
            {
                if (sqlite3PagerPageRefcount((ppPage).pDbPage) > 1)
                {
                    releasePage(ppPage);
                    return SQLITE_CORRUPT_BKPT();
                }
                (ppPage).isInit = 0;
            }
            else
            {
                ppPage = null;
            }
            Debug.Assert(rc != SQLITE_OK || sqlite3PagerIswriteable((ppPage).pDbPage));
            return rc;
        }

        /*
        ** This function is used to add page iPage to the database file free-list.
        ** It is assumed that the page is not already a part of the free-list.
        **
        ** The value passed as the second argument to this function is optional.
        ** If the caller happens to have a pointer to the MemPage object
        ** corresponding to page iPage handy, it may pass it as the second value.
        ** Otherwise, it may pass NULL.
        **
        ** If a pointer to a MemPage object is passed as the second argument,
        ** its reference count is not altered by this function.
        */
        static int freePage2(BtShared pBt, MemPage pMemPage, Pgno iPage)
        {
            MemPage pTrunk = null;                /* Free-list trunk page */
            Pgno iTrunk = 0;                      /* Page number of free-list trunk page */
            MemPage pPage1 = pBt.pPage1;          /* Local reference to page 1 */
            MemPage pPage;                        /* Page being freed. May be NULL. */
            int rc;                               /* Return Code */
            int nFree;                           /* Initial number of pages on free-list */

            Debug.Assert(sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(iPage > 1);
            Debug.Assert(null == pMemPage || pMemPage.pgno == iPage);

            if (pMemPage != null)
            {
                pPage = pMemPage;
                sqlite3PagerRef(pPage.pDbPage);
            }
            else
            {
                pPage = btreePageLookup(pBt, iPage);
            }

            /* Increment the free page count on pPage1 */
            rc = sqlite3PagerWrite(pPage1.pDbPage);
            if (rc != 0)
                goto freepage_out;
            nFree = (int)sqlite3Get4byte(pPage1.aData, 36);
            sqlite3Put4byte(pPage1.aData, 36, nFree + 1);

            if (pBt.secureDelete)
            {
                /* If the secure_delete option is enabled, then
                ** always fully overwrite deleted information with zeros.
                */
                if ((null == pPage && ((rc = btreeGetPage(pBt, iPage, ref pPage, 0)) != 0))
                || ((rc = sqlite3PagerWrite(pPage.pDbPage)) != 0)
                )
                {
                    goto freepage_out;
                }
                Array.Clear(pPage.aData, 0, (int)pPage.pBt.pageSize);//memset(pPage->aData, 0, pPage->pBt->pageSize);
            }

            /* If the database supports auto-vacuum, write an entry in the pointer-map
            ** to indicate that the page is free.
            */
#if !SQLITE_OMIT_AUTOVACUUM //   if ( ISAUTOVACUUM )
            if (pBt.autoVacuum)
#else
if (false)
#endif
            {
                ptrmapPut(pBt, iPage, PTRMAP_FREEPAGE, 0, ref rc);
                if (rc != 0)
                    goto freepage_out;
            }

            /* Now manipulate the actual database free-list structure. There are two
            ** possibilities. If the free-list is currently empty, or if the first
            ** trunk page in the free-list is full, then this page will become a
            ** new free-list trunk page. Otherwise, it will become a leaf of the
            ** first trunk page in the current free-list. This block tests if it
            ** is possible to add the page as a new free-list leaf.
            */
            if (nFree != 0)
            {
                u32 nLeaf;                /* Initial number of leaf cells on trunk page */

                iTrunk = sqlite3Get4byte(pPage1.aData, 32);
                rc = btreeGetPage(pBt, iTrunk, ref pTrunk, 0);
                if (rc != SQLITE_OK)
                {
                    goto freepage_out;
                }

                nLeaf = sqlite3Get4byte(pTrunk.aData, 4);
                Debug.Assert(pBt.usableSize > 32);
                if (nLeaf > (u32)pBt.usableSize / 4 - 2)
                {
                    rc = SQLITE_CORRUPT_BKPT();
                    goto freepage_out;
                }
                if (nLeaf < (u32)pBt.usableSize / 4 - 8)
                {
                    /* In this case there is room on the trunk page to insert the page
                    ** being freed as a new leaf.
                    **
                    ** Note that the trunk page is not really full until it contains
                    ** usableSize/4 - 2 entries, not usableSize/4 - 8 entries as we have
                    ** coded.  But due to a coding error in versions of SQLite prior to
                    ** 3.6.0, databases with freelist trunk pages holding more than
                    ** usableSize/4 - 8 entries will be reported as corrupt.  In order
                    ** to maintain backwards compatibility with older versions of SQLite,
                    ** we will continue to restrict the number of entries to usableSize/4 - 8
                    ** for now.  At some point in the future (once everyone has upgraded
                    ** to 3.6.0 or later) we should consider fixing the conditional above
                    ** to read "usableSize/4-2" instead of "usableSize/4-8".
                    */
                    rc = sqlite3PagerWrite(pTrunk.pDbPage);
                    if (rc == SQLITE_OK)
                    {
                        sqlite3Put4byte(pTrunk.aData, (u32)4, nLeaf + 1);
                        sqlite3Put4byte(pTrunk.aData, (u32)8 + nLeaf * 4, iPage);
                        if (pPage != null && !pBt.secureDelete)
                        {
                            sqlite3PagerDontWrite(pPage.pDbPage);
                        }
                        rc = btreeSetHasContent(pBt, iPage);
                    }
                    TRACE("FREE-PAGE: %d leaf on trunk page %d\n", iPage, pTrunk.pgno);
                    goto freepage_out;
                }
            }

            /* If control flows to this point, then it was not possible to add the
            ** the page being freed as a leaf page of the first trunk in the free-list.
            ** Possibly because the free-list is empty, or possibly because the
            ** first trunk in the free-list is full. Either way, the page being freed
            ** will become the new first trunk page in the free-list.
            */
            if (pPage == null && SQLITE_OK != (rc = btreeGetPage(pBt, iPage, ref pPage, 0)))
            {
                goto freepage_out;
            }
            rc = sqlite3PagerWrite(pPage.pDbPage);
            if (rc != SQLITE_OK)
            {
                goto freepage_out;
            }
            sqlite3Put4byte(pPage.aData, iTrunk);
            sqlite3Put4byte(pPage.aData, 4, 0);
            sqlite3Put4byte(pPage1.aData, (u32)32, iPage);
            TRACE("FREE-PAGE: %d new trunk page replacing %d\n", pPage.pgno, iTrunk);

        freepage_out:
            if (pPage != null)
            {
                pPage.isInit = 0;
            }
            releasePage(pPage);
            releasePage(pTrunk);
            return rc;
        }
        static void freePage(MemPage pPage, ref int pRC)
        {
            if ((pRC) == SQLITE_OK)
            {
                pRC = freePage2(pPage.pBt, pPage, pPage.pgno);
            }
        }

        /*
        ** Free any overflow pages associated with the given Cell.
        */
        static int clearCell(MemPage pPage, int pCell)
        {
            BtShared pBt = pPage.pBt;
            CellInfo info = new CellInfo();
            Pgno ovflPgno;
            int rc;
            int nOvfl;
            u32 ovflPageSize;

            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            btreeParseCellPtr(pPage, pCell, ref info);
            if (info.iOverflow == 0)
            {
                return SQLITE_OK;  /* No overflow pages. Return without doing anything */
            }
            ovflPgno = sqlite3Get4byte(pPage.aData, pCell, info.iOverflow);
            Debug.Assert(pBt.usableSize > 4);
            ovflPageSize = (u16)(pBt.usableSize - 4);
            nOvfl = (int)((info.nPayload - info.nLocal + ovflPageSize - 1) / ovflPageSize);
            Debug.Assert(ovflPgno == 0 || nOvfl > 0);
            while (nOvfl-- != 0)
            {
                Pgno iNext = 0;
                MemPage pOvfl = null;
                if (ovflPgno < 2 || ovflPgno > btreePagecount(pBt))
                {
                    /* 0 is not a legal page number and page 1 cannot be an
                    ** overflow page. Therefore if ovflPgno<2 or past the end of the
                    ** file the database must be corrupt. */
                    return SQLITE_CORRUPT_BKPT();
                }
                if (nOvfl != 0)
                {
                    rc = getOverflowPage(pBt, ovflPgno, out pOvfl, out iNext);
                    if (rc != 0)
                        return rc;
                }

                if ((pOvfl != null || ((pOvfl = btreePageLookup(pBt, ovflPgno)) != null))
                && sqlite3PagerPageRefcount(pOvfl.pDbPage) != 1
                )
                {
                    /* There is no reason any cursor should have an outstanding reference 
                    ** to an overflow page belonging to a cell that is being deleted/updated.
                    ** So if there exists more than one reference to this page, then it 
                    ** must not really be an overflow page and the database must be corrupt. 
                    ** It is helpful to detect this before calling freePage2(), as 
                    ** freePage2() may zero the page contents if secure-delete mode is
                    ** enabled. If this 'overflow' page happens to be a page that the
                    ** caller is iterating through or using in some other way, this
                    ** can be problematic.
                    */
                    rc = SQLITE_CORRUPT_BKPT();
                }
                else
                {
                    rc = freePage2(pBt, pOvfl, ovflPgno);
                }
                if (pOvfl != null)
                {
                    sqlite3PagerUnref(pOvfl.pDbPage);
                }
                if (rc != 0)
                    return rc;
                ovflPgno = iNext;
            }
            return SQLITE_OK;
        }

        /*
        ** Create the byte sequence used to represent a cell on page pPage
        ** and write that byte sequence into pCell[].  Overflow pages are
        ** allocated and filled in as necessary.  The calling procedure
        ** is responsible for making sure sufficient space has been allocated
        ** for pCell[].
        **
        ** Note that pCell does not necessary need to point to the pPage.aData
        ** area.  pCell might point to some temporary storage.  The cell will
        ** be constructed in this temporary area then copied into pPage.aData
        ** later.
        */
        static int fillInCell(
        MemPage pPage,            /* The page that contains the cell */
        byte[] pCell,             /* Complete text of the cell */
        byte[] pKey, i64 nKey,    /* The key */
        byte[] pData, int nData,  /* The data */
        int nZero,                /* Extra zero bytes to append to pData */
        ref int pnSize            /* Write cell size here */
        )
        {
            int nPayload;
            u8[] pSrc;
            int pSrcIndex = 0;
            int nSrc, n, rc;
            int spaceLeft;
            MemPage pOvfl = null;
            MemPage pToRelease = null;
            byte[] pPrior;
            int pPriorIndex = 0;
            byte[] pPayload;
            int pPayloadIndex = 0;
            BtShared pBt = pPage.pBt;
            Pgno pgnoOvfl = 0;
            int nHeader;
            CellInfo info = new CellInfo();

            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));

            /* pPage is not necessarily writeable since pCell might be auxiliary
            ** buffer space that is separate from the pPage buffer area */
            // TODO -- Determine if the following Assert is needed under c#
            //Debug.Assert( pCell < pPage.aData || pCell >= &pPage.aData[pBt.pageSize]
            //          || sqlite3PagerIswriteable(pPage.pDbPage) );

            /* Fill in the header. */
            nHeader = 0;
            if (0 == pPage.leaf)
            {
                nHeader += 4;
            }
            if (pPage.hasData != 0)
            {
                nHeader += (int)putVarint(pCell, nHeader, (int)(nData + nZero)); //putVarint( pCell[nHeader], nData + nZero );
            }
            else
            {
                nData = nZero = 0;
            }
            nHeader += putVarint(pCell, nHeader, (u64)nKey); //putVarint( pCell[nHeader], *(u64*)&nKey );
            btreeParseCellPtr(pPage, pCell, ref info);
            Debug.Assert(info.nHeader == nHeader);
            Debug.Assert(info.nKey == nKey);
            Debug.Assert(info.nData == (u32)(nData + nZero));

            /* Fill in the payload */
            nPayload = nData + nZero;
            if (pPage.intKey != 0)
            {
                pSrc = pData;
                nSrc = nData;
                nData = 0;
            }
            else
            {
                if (NEVER(nKey > 0x7fffffff || pKey == null))
                {
                    return SQLITE_CORRUPT_BKPT();
                }
                nPayload += (int)nKey;
                pSrc = pKey;
                nSrc = (int)nKey;
            }
            pnSize = info.nSize;
            spaceLeft = info.nLocal;
            //  pPayload = &pCell[nHeader];
            pPayload = pCell;
            pPayloadIndex = nHeader;
            //  pPrior = &pCell[info.iOverflow];
            pPrior = pCell;
            pPriorIndex = info.iOverflow;

            while (nPayload > 0)
            {
                if (spaceLeft == 0)
                {
#if !SQLITE_OMIT_AUTOVACUUM
                    Pgno pgnoPtrmap = pgnoOvfl; /* Overflow page pointer-map entry page */
                    if (pBt.autoVacuum)
                    {
                        do
                        {
                            pgnoOvfl++;
                        } while (
                        PTRMAP_ISPAGE(pBt, pgnoOvfl) || pgnoOvfl == PENDING_BYTE_PAGE(pBt)
                        );
                    }
#endif
                    rc = allocateBtreePage(pBt, ref pOvfl, ref pgnoOvfl, pgnoOvfl, 0);
#if !SQLITE_OMIT_AUTOVACUUM
                    /* If the database supports auto-vacuum, and the second or subsequent
** overflow page is being allocated, add an entry to the pointer-map
** for that page now.
**
** If this is the first overflow page, then write a partial entry
** to the pointer-map. If we write nothing to this pointer-map slot,
** then the optimistic overflow chain processing in clearCell()
** may misinterpret the uninitialised values and delete the
** wrong pages from the database.
*/
                    if (pBt.autoVacuum && rc == SQLITE_OK)
                    {
                        u8 eType = (u8)(pgnoPtrmap != 0 ? PTRMAP_OVERFLOW2 : PTRMAP_OVERFLOW1);
                        ptrmapPut(pBt, pgnoOvfl, eType, pgnoPtrmap, ref rc);
                        if (rc != 0)
                        {
                            releasePage(pOvfl);
                        }
                    }
#endif
                    if (rc != 0)
                    {
                        releasePage(pToRelease);
                        return rc;
                    }

                    /* If pToRelease is not zero than pPrior points into the data area
                    ** of pToRelease.  Make sure pToRelease is still writeable. */
                    Debug.Assert(pToRelease == null || sqlite3PagerIswriteable(pToRelease.pDbPage));

                    /* If pPrior is part of the data area of pPage, then make sure pPage
                    ** is still writeable */
                    // TODO -- Determine if the following Assert is needed under c#
                    //Debug.Assert( pPrior < pPage.aData || pPrior >= &pPage.aData[pBt.pageSize]
                    //      || sqlite3PagerIswriteable(pPage.pDbPage) );

                    sqlite3Put4byte(pPrior, pPriorIndex, pgnoOvfl);
                    releasePage(pToRelease);
                    pToRelease = pOvfl;
                    pPrior = pOvfl.aData;
                    pPriorIndex = 0;
                    sqlite3Put4byte(pPrior, 0);
                    pPayload = pOvfl.aData;
                    pPayloadIndex = 4; //&pOvfl.aData[4];
                    spaceLeft = (int)pBt.usableSize - 4;
                }
                n = nPayload;
                if (n > spaceLeft)
                    n = spaceLeft;

                /* If pToRelease is not zero than pPayload points into the data area
                ** of pToRelease.  Make sure pToRelease is still writeable. */
                Debug.Assert(pToRelease == null || sqlite3PagerIswriteable(pToRelease.pDbPage));

                /* If pPayload is part of the data area of pPage, then make sure pPage
                ** is still writeable */
                // TODO -- Determine if the following Assert is needed under c#
                //Debug.Assert( pPayload < pPage.aData || pPayload >= &pPage.aData[pBt.pageSize]
                //        || sqlite3PagerIswriteable(pPage.pDbPage) );

                if (nSrc > 0)
                {
                    if (n > nSrc)
                        n = nSrc;
                    Debug.Assert(pSrc != null);
                    Buffer.BlockCopy(pSrc, pSrcIndex, pPayload, pPayloadIndex, n);//memcpy(pPayload, pSrc, n);
                }
                else
                {
                    byte[] pZeroBlob = sqlite3Malloc(n); // memset(pPayload, 0, n);
                    Buffer.BlockCopy(pZeroBlob, 0, pPayload, pPayloadIndex, n);
                }
                nPayload -= n;
                pPayloadIndex += n;// pPayload += n;
                pSrcIndex += n;// pSrc += n;
                nSrc -= n;
                spaceLeft -= n;
                if (nSrc == 0)
                {
                    nSrc = nData;
                    pSrc = pData;
                }
            }
            releasePage(pToRelease);
            return SQLITE_OK;
        }

        /*
        ** Remove the i-th cell from pPage.  This routine effects pPage only.
        ** The cell content is not freed or deallocated.  It is assumed that
        ** the cell content has been copied someplace else.  This routine just
        ** removes the reference to the cell from pPage.
        **
        ** "sz" must be the number of bytes in the cell.
        */
        static void dropCell(MemPage pPage, int idx, int sz, ref int pRC)
        {
            u32 pc;         /* Offset to cell content of cell being deleted */
            u8[] data;      /* pPage.aData */
            int ptr;        /* Used to move bytes around within data[] */
            int endPtr;     /* End of loop */
            int rc;         /* The return code */
            int hdr;        /* Beginning of the header.  0 most pages.  100 page 1 */

            if (pRC != 0)
                return;

            Debug.Assert(idx >= 0 && idx < pPage.nCell);
#if SQLITE_DEBUG
  Debug.Assert( sz == cellSize( pPage, idx ) );
#endif
            Debug.Assert(sqlite3PagerIswriteable(pPage.pDbPage));
            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            data = pPage.aData;
            ptr = pPage.cellOffset + 2 * idx; //ptr = &data[pPage.cellOffset + 2 * idx];
            pc = (u32)get2byte(data, ptr);
            hdr = pPage.hdrOffset;
            testcase(pc == get2byte(data, hdr + 5));
            testcase(pc + sz == pPage.pBt.usableSize);
            if (pc < (u32)get2byte(data, hdr + 5) || pc + sz > pPage.pBt.usableSize)
            {
                pRC = SQLITE_CORRUPT_BKPT();
                return;
            }
            rc = freeSpace(pPage, pc, sz);
            if (rc != 0)
            {
                pRC = rc;
                return;
            }
            //endPtr = &data[pPage->cellOffset + 2*pPage->nCell - 2];
            //assert( (SQLITE_PTR_TO_INT(ptr)&1)==0 );  /* ptr is always 2-byte aligned */
            //while( ptr<endPtr ){
            //  *(u16*)ptr = *(u16*)&ptr[2];
            //  ptr += 2;
            Buffer.BlockCopy(data, ptr + 2, data, ptr, (pPage.nCell - 1 - idx) * 2);
            pPage.nCell--;
            data[pPage.hdrOffset + 3] = (byte)(pPage.nCell >> 8);
            data[pPage.hdrOffset + 4] = (byte)(pPage.nCell); //put2byte( data, hdr + 3, pPage.nCell );
            pPage.nFree += 2;
        }

        /*
        ** Insert a new cell on pPage at cell index "i".  pCell points to the
        ** content of the cell.
        **
        ** If the cell content will fit on the page, then put it there.  If it
        ** will not fit, then make a copy of the cell content into pTemp if
        ** pTemp is not null.  Regardless of pTemp, allocate a new entry
        ** in pPage.aOvfl[] and make it point to the cell content (either
        ** in pTemp or the original pCell) and also record its index.
        ** Allocating a new entry in pPage.aCell[] implies that
        ** pPage.nOverflow is incremented.
        **
        ** If nSkip is non-zero, then do not copy the first nSkip bytes of the
        ** cell. The caller will overwrite them after this function returns. If
        ** nSkip is non-zero, then pCell may not point to an invalid memory location
        ** (but pCell+nSkip is always valid).
        */
        static void insertCell(
        MemPage pPage,      /* Page into which we are copying */
        int i,              /* New cell becomes the i-th cell of the page */
        u8[] pCell,         /* Content of the new cell */
        int sz,             /* Bytes of content in pCell */
        u8[] pTemp,         /* Temp storage space for pCell, if needed */
        Pgno iChild,        /* If non-zero, replace first 4 bytes with this value */
        ref int pRC         /* Read and write return code from here */
        )
        {
            int idx = 0;      /* Where to write new cell content in data[] */
            int j;            /* Loop counter */
            int end;          /* First byte past the last cell pointer in data[] */
            int ins;          /* Index in data[] where new cell pointer is inserted */
            int cellOffset;   /* Address of first cell pointer in data[] */
            u8[] data;        /* The content of the whole page */
            u8 ptr;           /* Used for moving information around in data[] */
            u8 endPtr;        /* End of the loop */

            int nSkip = (iChild != 0 ? 4 : 0);

            if (pRC != 0)
                return;

            Debug.Assert(i >= 0 && i <= pPage.nCell + pPage.nOverflow);
            Debug.Assert(pPage.nCell <= MX_CELL(pPage.pBt) && MX_CELL(pPage.pBt) <= 10921);
            Debug.Assert(pPage.nOverflow <= ArraySize(pPage.aOvfl));
            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            /* The cell should normally be sized correctly.  However, when moving a
            ** malformed cell from a leaf page to an interior page, if the cell size
            ** wanted to be less than 4 but got rounded up to 4 on the leaf, then size
            ** might be less than 8 (leaf-size + pointer) on the interior node.  Hence
            ** the term after the || in the following assert(). */
            Debug.Assert(sz == cellSizePtr(pPage, pCell) || (sz == 8 && iChild > 0));
            if (pPage.nOverflow != 0 || sz + 2 > pPage.nFree)
            {
                if (pTemp != null)
                {
                    Buffer.BlockCopy(pCell, nSkip, pTemp, nSkip, sz - nSkip);//memcpy(pTemp+nSkip, pCell+nSkip, sz-nSkip);
                    pCell = pTemp;
                }
                if (iChild != 0)
                {
                    sqlite3Put4byte(pCell, iChild);
                }
                j = pPage.nOverflow++;
                Debug.Assert(j < pPage.aOvfl.Length);//(int)(sizeof(pPage.aOvfl)/sizeof(pPage.aOvfl[0])) );
                pPage.aOvfl[j].pCell = pCell;
                pPage.aOvfl[j].idx = (u16)i;
            }
            else
            {
                int rc = sqlite3PagerWrite(pPage.pDbPage);
                if (rc != SQLITE_OK)
                {
                    pRC = rc;
                    return;
                }
                Debug.Assert(sqlite3PagerIswriteable(pPage.pDbPage));
                data = pPage.aData;
                cellOffset = pPage.cellOffset;
                end = cellOffset + 2 * pPage.nCell;
                ins = cellOffset + 2 * i;
                rc = allocateSpace(pPage, sz, ref idx);
                if (rc != 0)
                {
                    pRC = rc;
                    return;
                }
                /* The allocateSpace() routine guarantees the following two properties
                ** if it returns success */
                Debug.Assert(idx >= end + 2);
                Debug.Assert(idx + sz <= (int)pPage.pBt.usableSize);
                pPage.nCell++;
                pPage.nFree -= (u16)(2 + sz);
                Buffer.BlockCopy(pCell, nSkip, data, idx + nSkip, sz - nSkip); //memcpy( data[idx + nSkip], pCell + nSkip, sz - nSkip );
                if (iChild != 0)
                {
                    sqlite3Put4byte(data, idx, iChild);
                }
                //ptr = &data[end];
                //endPtr = &data[ins];
                //assert( ( SQLITE_PTR_TO_INT( ptr ) & 1 ) == 0 );  /* ptr is always 2-byte aligned */
                //while ( ptr > endPtr )
                //{
                //  *(u16*)ptr = *(u16*)&ptr[-2];
                //  ptr -= 2;
                //}
                for (j = end; j > ins; j -= 2)
                {
                    data[j + 0] = data[j - 2];
                    data[j + 1] = data[j - 1];
                }
                put2byte(data, ins, idx);
                put2byte(data, pPage.hdrOffset + 3, pPage.nCell);
#if !SQLITE_OMIT_AUTOVACUUM
                if (pPage.pBt.autoVacuum)
                {
                    /* The cell may contain a pointer to an overflow page. If so, write
                    ** the entry for the overflow page into the pointer map.
                    */
                    ptrmapPutOvflPtr(pPage, pCell, ref pRC);
                }
#endif
            }
        }

        /*
        ** Add a list of cells to a page.  The page should be initially empty.
        ** The cells are guaranteed to fit on the page.
        */
        static void assemblePage(
        MemPage pPage,    /* The page to be assemblied */
        int nCell,        /* The number of cells to add to this page */
        u8[] apCell,      /* Pointer to a single the cell bodies */
        int[] aSize       /* Sizes of the cells bodie*/
        )
        {
            int i;            /* Loop counter */
            int pCellptr;     /* Address of next cell pointer */
            int cellbody;     /* Address of next cell body */
            byte[] data = pPage.aData;          /* Pointer to data for pPage */
            int hdr = pPage.hdrOffset;          /* Offset of header on pPage */
            int nUsable = (int)pPage.pBt.usableSize; /* Usable size of page */

            Debug.Assert(pPage.nOverflow == 0);
            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(nCell >= 0 && nCell <= (int)MX_CELL(pPage.pBt)
                  && (int)MX_CELL(pPage.pBt) <= 10921);

            Debug.Assert(sqlite3PagerIswriteable(pPage.pDbPage));

            /* Check that the page has just been zeroed by zeroPage() */
            Debug.Assert(pPage.nCell == 0);
            Debug.Assert(get2byteNotZero(data, hdr + 5) == nUsable);

            pCellptr = pPage.cellOffset + nCell * 2; //data[pPage.cellOffset + nCell * 2];
            cellbody = nUsable;
            for (i = nCell - 1; i >= 0; i--)
            {
                u16 sz = (u16)aSize[i];
                pCellptr -= 2;
                cellbody -= sz;
                put2byte(data, pCellptr, cellbody);
                Buffer.BlockCopy(apCell, 0, data, cellbody, sz);// memcpy(&data[cellbody], apCell[i], sz);
            }
            put2byte(data, hdr + 3, nCell);
            put2byte(data, hdr + 5, cellbody);
            pPage.nFree -= (u16)(nCell * 2 + nUsable - cellbody);
            pPage.nCell = (u16)nCell;
        }
        static void assemblePage(
        MemPage pPage,    /* The page to be assemblied */
        int nCell,        /* The number of cells to add to this page */
        u8[][] apCell,    /* Pointers to cell bodies */
        u16[] aSize,      /* Sizes of the cells */
        int offset        /* Offset into the cell bodies, for c#  */
        )
        {
            int i;            /* Loop counter */
            int pCellptr;      /* Address of next cell pointer */
            int cellbody;     /* Address of next cell body */
            byte[] data = pPage.aData;          /* Pointer to data for pPage */
            int hdr = pPage.hdrOffset;          /* Offset of header on pPage */
            int nUsable = (int)pPage.pBt.usableSize; /* Usable size of page */

            Debug.Assert(pPage.nOverflow == 0);
            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(nCell >= 0 && nCell <= MX_CELL(pPage.pBt) && MX_CELL(pPage.pBt) <= 5460);
            Debug.Assert(sqlite3PagerIswriteable(pPage.pDbPage));

            /* Check that the page has just been zeroed by zeroPage() */
            Debug.Assert(pPage.nCell == 0);
            Debug.Assert(get2byte(data, hdr + 5) == nUsable);

            pCellptr = pPage.cellOffset + nCell * 2; //data[pPage.cellOffset + nCell * 2];
            cellbody = nUsable;
            for (i = nCell - 1; i >= 0; i--)
            {
                pCellptr -= 2;
                cellbody -= aSize[i + offset];
                put2byte(data, pCellptr, cellbody);
                Buffer.BlockCopy(apCell[offset + i], 0, data, cellbody, aSize[i + offset]);//          memcpy(&data[cellbody], apCell[i], aSize[i]);
            }
            put2byte(data, hdr + 3, nCell);
            put2byte(data, hdr + 5, cellbody);
            pPage.nFree -= (u16)(nCell * 2 + nUsable - cellbody);
            pPage.nCell = (u16)nCell;
        }

        static void assemblePage(
        MemPage pPage,    /* The page to be assemblied */
        int nCell,        /* The number of cells to add to this page */
        u8[] apCell,      /* Pointers to cell bodies */
        u16[] aSize       /* Sizes of the cells */
        )
        {
            int i;            /* Loop counter */
            int pCellptr;     /* Address of next cell pointer */
            int cellbody;     /* Address of next cell body */
            u8[] data = pPage.aData;             /* Pointer to data for pPage */
            int hdr = pPage.hdrOffset;           /* Offset of header on pPage */
            int nUsable = (int)pPage.pBt.usableSize; /* Usable size of page */

            Debug.Assert(pPage.nOverflow == 0);
            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(nCell >= 0 && nCell <= MX_CELL(pPage.pBt) && MX_CELL(pPage.pBt) <= 5460);
            Debug.Assert(sqlite3PagerIswriteable(pPage.pDbPage));

            /* Check that the page has just been zeroed by zeroPage() */
            Debug.Assert(pPage.nCell == 0);
            Debug.Assert(get2byte(data, hdr + 5) == nUsable);

            pCellptr = pPage.cellOffset + nCell * 2; //&data[pPage.cellOffset + nCell * 2];
            cellbody = nUsable;
            for (i = nCell - 1; i >= 0; i--)
            {
                pCellptr -= 2;
                cellbody -= aSize[i];
                put2byte(data, pCellptr, cellbody);
                Buffer.BlockCopy(apCell, 0, data, cellbody, aSize[i]);//memcpy( data[cellbody], apCell[i], aSize[i] );
            }
            put2byte(data, hdr + 3, nCell);
            put2byte(data, hdr + 5, cellbody);
            pPage.nFree -= (u16)(nCell * 2 + nUsable - cellbody);
            pPage.nCell = (u16)nCell;
        }

        /*
        ** The following parameters determine how many adjacent pages get involved
        ** in a balancing operation.  NN is the number of neighbors on either side
        ** of the page that participate in the balancing operation.  NB is the
        ** total number of pages that participate, including the target page and
        ** NN neighbors on either side.
        **
        ** The minimum value of NN is 1 (of course).  Increasing NN above 1
        ** (to 2 or 3) gives a modest improvement in SELECT and DELETE performance
        ** in exchange for a larger degradation in INSERT and UPDATE performance.
        ** The value of NN appears to give the best results overall.
        */
        static int NN = 1;              /* Number of neighbors on either side of pPage */
        static int NB = (NN * 2 + 1);   /* Total pages involved in the balance */

#if !SQLITE_OMIT_QUICKBALANCE
        /*
** This version of balance() handles the common special case where
** a new entry is being inserted on the extreme right-end of the
** tree, in other words, when the new entry will become the largest
** entry in the tree.
**
** Instead of trying to balance the 3 right-most leaf pages, just add
** a new page to the right-hand side and put the one new entry in
** that page.  This leaves the right side of the tree somewhat
** unbalanced.  But odds are that we will be inserting new entries
** at the end soon afterwards so the nearly empty page will quickly
** fill up.  On average.
**
** pPage is the leaf page which is the right-most page in the tree.
** pParent is its parent.  pPage must have a single overflow entry
** which is also the right-most entry on the page.
**
** The pSpace buffer is used to store a temporary copy of the divider
** cell that will be inserted into pParent. Such a cell consists of a 4
** byte page number followed by a variable length integer. In other
** words, at most 13 bytes. Hence the pSpace buffer must be at
** least 13 bytes in size.
*/
        static int balance_quick(MemPage pParent, MemPage pPage, u8[] pSpace)
        {
            BtShared pBt = pPage.pBt;    /* B-Tree Database */
            MemPage pNew = new MemPage();/* Newly allocated page */
            int rc;                      /* Return Code */
            Pgno pgnoNew = 0;              /* Page number of pNew */

            Debug.Assert(sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(sqlite3PagerIswriteable(pParent.pDbPage));
            Debug.Assert(pPage.nOverflow == 1);

            /* This error condition is now caught prior to reaching this function */
            if (pPage.nCell <= 0)
                return SQLITE_CORRUPT_BKPT();

            /* Allocate a new page. This page will become the right-sibling of
            ** pPage. Make the parent page writable, so that the new divider cell
            ** may be inserted. If both these operations are successful, proceed.
            */
            rc = allocateBtreePage(pBt, ref pNew, ref pgnoNew, 0, 0);

            if (rc == SQLITE_OK)
            {

                int pOut = 4;//u8 pOut = &pSpace[4];
                u8[] pCell = pPage.aOvfl[0].pCell;
                int[] szCell = new int[1];
                szCell[0] = cellSizePtr(pPage, pCell);
                int pStop;

                Debug.Assert(sqlite3PagerIswriteable(pNew.pDbPage));
                Debug.Assert(pPage.aData[0] == (PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF));
                zeroPage(pNew, PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF);
                assemblePage(pNew, 1, pCell, szCell);

                /* If this is an auto-vacuum database, update the pointer map
                ** with entries for the new page, and any pointer from the
                ** cell on the page to an overflow page. If either of these
                ** operations fails, the return code is set, but the contents
                ** of the parent page are still manipulated by thh code below.
                ** That is Ok, at this point the parent page is guaranteed to
                ** be marked as dirty. Returning an error code will cause a
                ** rollback, undoing any changes made to the parent page.
                */
#if !SQLITE_OMIT_AUTOVACUUM //   if ( ISAUTOVACUUM )
                if (pBt.autoVacuum)
#else
if (false)
#endif
                {
                    ptrmapPut(pBt, pgnoNew, PTRMAP_BTREE, pParent.pgno, ref rc);
                    if (szCell[0] > pNew.minLocal)
                    {
                        ptrmapPutOvflPtr(pNew, pCell, ref rc);
                    }
                }

                /* Create a divider cell to insert into pParent. The divider cell
                ** consists of a 4-byte page number (the page number of pPage) and
                ** a variable length key value (which must be the same value as the
                ** largest key on pPage).
                **
                ** To find the largest key value on pPage, first find the right-most
                ** cell on pPage. The first two fields of this cell are the
                ** record-length (a variable length integer at most 32-bits in size)
                ** and the key value (a variable length integer, may have any value).
                ** The first of the while(...) loops below skips over the record-length
                ** field. The second while(...) loop copies the key value from the
                ** cell on pPage into the pSpace buffer.
                */
                int iCell = findCell(pPage, pPage.nCell - 1); //pCell = findCell( pPage, pPage.nCell - 1 );
                pCell = pPage.aData;
                int _pCell = iCell;
                pStop = _pCell + 9; //pStop = &pCell[9];
                while (((pCell[_pCell++]) & 0x80) != 0 && _pCell < pStop)
                    ; //while ( ( *( pCell++ ) & 0x80 ) && pCell < pStop ) ;
                pStop = _pCell + 9;//pStop = &pCell[9];
                while (((pSpace[pOut++] = pCell[_pCell++]) & 0x80) != 0 && _pCell < pStop)
                    ; //while ( ( ( *( pOut++ ) = *( pCell++ ) ) & 0x80 ) && pCell < pStop ) ;

                /* Insert the new divider cell into pParent. */
                insertCell(pParent, pParent.nCell, pSpace, pOut, //(int)(pOut-pSpace),
                null, pPage.pgno, ref rc);

                /* Set the right-child pointer of pParent to point to the new page. */
                sqlite3Put4byte(pParent.aData, pParent.hdrOffset + 8, pgnoNew);

                /* Release the reference to the new page. */
                releasePage(pNew);
            }

            return rc;
        }
#endif //* SQLITE_OMIT_QUICKBALANCE */

#if FALSE
/*
** This function does not contribute anything to the operation of SQLite.
** it is sometimes activated temporarily while debugging code responsible
** for setting pointer-map entries.
*/
static int ptrmapCheckPages(MemPage **apPage, int nPage){
int i, j;
for(i=0; i<nPage; i++){
Pgno n;
u8 e;
MemPage pPage = apPage[i];
BtShared pBt = pPage.pBt;
Debug.Assert( pPage.isInit!=0 );

for(j=0; j<pPage.nCell; j++){
CellInfo info;
u8 *z;

z = findCell(pPage, j);
btreeParseCellPtr(pPage, z,  info);
if( info.iOverflow ){
Pgno ovfl = sqlite3Get4byte(z[info.iOverflow]);
ptrmapGet(pBt, ovfl, ref e, ref n);
Debug.Assert( n==pPage.pgno && e==PTRMAP_OVERFLOW1 );
}
if( 0==pPage.leaf ){
Pgno child = sqlite3Get4byte(z);
ptrmapGet(pBt, child, ref e, ref n);
Debug.Assert( n==pPage.pgno && e==PTRMAP_BTREE );
}
}
if( 0==pPage.leaf ){
Pgno child = sqlite3Get4byte(pPage.aData,pPage.hdrOffset+8]);
ptrmapGet(pBt, child, ref e, ref n);
Debug.Assert( n==pPage.pgno && e==PTRMAP_BTREE );
}
}
return 1;
}
#endif

        /*
** This function is used to copy the contents of the b-tree node stored
** on page pFrom to page pTo. If page pFrom was not a leaf page, then
** the pointer-map entries for each child page are updated so that the
** parent page stored in the pointer map is page pTo. If pFrom contained
** any cells with overflow page pointers, then the corresponding pointer
** map entries are also updated so that the parent page is page pTo.
**
** If pFrom is currently carrying any overflow cells (entries in the
** MemPage.aOvfl[] array), they are not copied to pTo.
**
** Before returning, page pTo is reinitialized using btreeInitPage().
**
** The performance of this function is not critical. It is only used by
** the balance_shallower() and balance_deeper() procedures, neither of
** which are called often under normal circumstances.
*/
        static void copyNodeContent(MemPage pFrom, MemPage pTo, ref int pRC)
        {
            if ((pRC) == SQLITE_OK)
            {
                BtShared pBt = pFrom.pBt;
                u8[] aFrom = pFrom.aData;
                u8[] aTo = pTo.aData;
                int iFromHdr = pFrom.hdrOffset;
                int iToHdr = ((pTo.pgno == 1) ? 100 : 0);
                int rc;
                int iData;


                Debug.Assert(pFrom.isInit != 0);
                Debug.Assert(pFrom.nFree >= iToHdr);
                Debug.Assert(get2byte(aFrom, iFromHdr + 5) <= (int)pBt.usableSize);

                /* Copy the b-tree node content from page pFrom to page pTo. */
                iData = get2byte(aFrom, iFromHdr + 5);
                Buffer.BlockCopy(aFrom, iData, aTo, iData, (int)pBt.usableSize - iData);//memcpy(aTo[iData], ref aFrom[iData], pBt.usableSize-iData);
                Buffer.BlockCopy(aFrom, iFromHdr, aTo, iToHdr, pFrom.cellOffset + 2 * pFrom.nCell);//memcpy(aTo[iToHdr], ref aFrom[iFromHdr], pFrom.cellOffset + 2*pFrom.nCell);

                /* Reinitialize page pTo so that the contents of the MemPage structure
                ** match the new data. The initialization of pTo can actually fail under
                ** fairly obscure circumstances, even though it is a copy of initialized 
                ** page pFrom.
                */
                pTo.isInit = 0;
                rc = btreeInitPage(pTo);
                if (rc != SQLITE_OK)
                {
                    pRC = rc;
                    return;
                }

                /* If this is an auto-vacuum database, update the pointer-map entries
                ** for any b-tree or overflow pages that pTo now contains the pointers to.
                */
#if !SQLITE_OMIT_AUTOVACUUM //   if ( ISAUTOVACUUM )
                if (pBt.autoVacuum)
#else
if (false)
#endif
                {
                    pRC = setChildPtrmaps(pTo);
                }
            }
        }

        /*
        ** This routine redistributes cells on the iParentIdx'th child of pParent
        ** (hereafter "the page") and up to 2 siblings so that all pages have about the
        ** same amount of free space. Usually a single sibling on either side of the
        ** page are used in the balancing, though both siblings might come from one
        ** side if the page is the first or last child of its parent. If the page
        ** has fewer than 2 siblings (something which can only happen if the page
        ** is a root page or a child of a root page) then all available siblings
        ** participate in the balancing.
        **
        ** The number of siblings of the page might be increased or decreased by
        ** one or two in an effort to keep pages nearly full but not over full.
        **
        ** Note that when this routine is called, some of the cells on the page
        ** might not actually be stored in MemPage.aData[]. This can happen
        ** if the page is overfull. This routine ensures that all cells allocated
        ** to the page and its siblings fit into MemPage.aData[] before returning.
        **
        ** In the course of balancing the page and its siblings, cells may be
        ** inserted into or removed from the parent page (pParent). Doing so
        ** may cause the parent page to become overfull or underfull. If this
        ** happens, it is the responsibility of the caller to invoke the correct
        ** balancing routine to fix this problem (see the balance() routine).
        **
        ** If this routine fails for any reason, it might leave the database
        ** in a corrupted state. So if this routine fails, the database should
        ** be rolled back.
        **
        ** The third argument to this function, aOvflSpace, is a pointer to a
        ** buffer big enough to hold one page. If while inserting cells into the parent
        ** page (pParent) the parent page becomes overfull, this buffer is
        ** used to store the parent's overflow cells. Because this function inserts
        ** a maximum of four divider cells into the parent page, and the maximum
        ** size of a cell stored within an internal node is always less than 1/4
        ** of the page-size, the aOvflSpace[] buffer is guaranteed to be large
        ** enough for all overflow cells.
        **
        ** If aOvflSpace is set to a null pointer, this function returns
        ** SQLITE_NOMEM.
        */

        // under C#; Try to reuse Memory

        static int balance_nonroot(
        MemPage pParent,               /* Parent page of siblings being balanced */
        int iParentIdx,                /* Index of "the page" in pParent */
        u8[] aOvflSpace,               /* page-size bytes of space for parent ovfl */
        int isRoot                     /* True if pParent is a root-page */
        )
        {
            MemPage[] apOld = new MemPage[NB];    /* pPage and up to two siblings */
            MemPage[] apCopy = new MemPage[NB];   /* Private copies of apOld[] pages */
            MemPage[] apNew = new MemPage[NB + 2];/* pPage and up to NB siblings after balancing */
            int[] apDiv = new int[NB - 1];        /* Divider cells in pParent */
            int[] cntNew = new int[NB + 2];       /* Index in aCell[] of cell after i-th page */
            int[] szNew = new int[NB + 2];        /* Combined size of cells place on i-th page */
            u16[] szCell = new u16[1];            /* Local size of all cells in apCell[] */
            BtShared pBt;                /* The whole database */
            int nCell = 0;               /* Number of cells in apCell[] */
            int nMaxCells = 0;           /* Allocated size of apCell, szCell, aFrom. */
            int nNew = 0;                /* Number of pages in apNew[] */
            int nOld;                    /* Number of pages in apOld[] */
            int i, j, k;                 /* Loop counters */
            int nxDiv;                   /* Next divider slot in pParent.aCell[] */
            int rc = SQLITE_OK;          /* The return code */
            u16 leafCorrection;          /* 4 if pPage is a leaf.  0 if not */
            int leafData;                /* True if pPage is a leaf of a LEAFDATA tree */
            int usableSpace;             /* Bytes in pPage beyond the header */
            int pageFlags;               /* Value of pPage.aData[0] */
            int subtotal;                /* Subtotal of bytes in cells on one page */
            //int iSpace1 = 0;             /* First unused byte of aSpace1[] */
            int iOvflSpace = 0;          /* First unused byte of aOvflSpace[] */
            int szScratch;               /* Size of scratch memory requested */
            int pRight;                  /* Location in parent of right-sibling pointer */
            u8[][] apCell = null;                 /* All cells begin balanced */
            //u16[] szCell;                         /* Local size of all cells in apCell[] */
            //u8[] aSpace1;                         /* Space for copies of dividers cells */
            Pgno pgno;                   /* Temp var to store a page number in */

            pBt = pParent.pBt;
            Debug.Assert(sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(sqlite3PagerIswriteable(pParent.pDbPage));

#if FALSE
TRACE("BALANCE: begin page %d child of %d\n", pPage.pgno, pParent.pgno);
#endif

            /* At this point pParent may have at most one overflow cell. And if
** this overflow cell is present, it must be the cell with
** index iParentIdx. This scenario comes about when this function
** is called (indirectly) from sqlite3BtreeDelete().
*/
            Debug.Assert(pParent.nOverflow == 0 || pParent.nOverflow == 1);
            Debug.Assert(pParent.nOverflow == 0 || pParent.aOvfl[0].idx == iParentIdx);

            //if( !aOvflSpace ){
            //  return SQLITE_NOMEM;
            //}

            /* Find the sibling pages to balance. Also locate the cells in pParent
            ** that divide the siblings. An attempt is made to find NN siblings on
            ** either side of pPage. More siblings are taken from one side, however,
            ** if there are fewer than NN siblings on the other side. If pParent
            ** has NB or fewer children then all children of pParent are taken.
            **
            ** This loop also drops the divider cells from the parent page. This
            ** way, the remainder of the function does not have to deal with any
            ** overflow cells in the parent page, since if any existed they will
            ** have already been removed.
            */
            i = pParent.nOverflow + pParent.nCell;
            if (i < 2)
            {
                nxDiv = 0;
                nOld = i + 1;
            }
            else
            {
                nOld = 3;
                if (iParentIdx == 0)
                {
                    nxDiv = 0;
                }
                else if (iParentIdx == i)
                {
                    nxDiv = i - 2;
                }
                else
                {
                    nxDiv = iParentIdx - 1;
                }
                i = 2;
            }
            if ((i + nxDiv - pParent.nOverflow) == pParent.nCell)
            {
                pRight = pParent.hdrOffset + 8; //&pParent.aData[pParent.hdrOffset + 8];
            }
            else
            {
                pRight = findCell(pParent, i + nxDiv - pParent.nOverflow);
            }
            pgno = sqlite3Get4byte(pParent.aData, pRight);
            while (true)
            {
                rc = getAndInitPage(pBt, pgno, ref apOld[i]);
                if (rc != 0)
                {
                    //memset(apOld, 0, (i+1)*sizeof(MemPage*));
                    goto balance_cleanup;
                }
                nMaxCells += 1 + apOld[i].nCell + apOld[i].nOverflow;
                if ((i--) == 0)
                    break;

                if (i + nxDiv == pParent.aOvfl[0].idx && pParent.nOverflow != 0)
                {
                    apDiv[i] = 0;// = pParent.aOvfl[0].pCell;
                    pgno = sqlite3Get4byte(pParent.aOvfl[0].pCell, apDiv[i]);
                    szNew[i] = cellSizePtr(pParent, apDiv[i]);
                    pParent.nOverflow = 0;
                }
                else
                {
                    apDiv[i] = findCell(pParent, i + nxDiv - pParent.nOverflow);
                    pgno = sqlite3Get4byte(pParent.aData, apDiv[i]);
                    szNew[i] = cellSizePtr(pParent, apDiv[i]);

                    /* Drop the cell from the parent page. apDiv[i] still points to
                    ** the cell within the parent, even though it has been dropped.
                    ** This is safe because dropping a cell only overwrites the first
                    ** four bytes of it, and this function does not need the first
                    ** four bytes of the divider cell. So the pointer is safe to use
                    ** later on.
                    **
                    ** Unless SQLite is compiled in secure-delete mode. In this case,
                    ** the dropCell() routine will overwrite the entire cell with zeroes.
                    ** In this case, temporarily copy the cell into the aOvflSpace[]
                    ** buffer. It will be copied out again as soon as the aSpace[] buffer
                    ** is allocated.  */
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
                    dropCell(pParent, i + nxDiv - pParent.nOverflow, szNew[i], ref rc);
                }
            }

            /* Make nMaxCells a multiple of 4 in order to preserve 8-byte
            ** alignment */
            nMaxCells = (nMaxCells + 3) & ~3;

            /*
            ** Allocate space for memory structures
            */
            //k = pBt.pageSize + ROUND8(sizeof(MemPage));
            //szScratch =
            //     nMaxCells*sizeof(u8*)                       /* apCell */
            //   + nMaxCells*sizeof(u16)                       /* szCell */
            //   + pBt.pageSize                               /* aSpace1 */
            //   + k*nOld;                                     /* Page copies (apCopy) */
            apCell = sqlite3ScratchMalloc(apCell, nMaxCells);
            //if( apCell==null ){
            //  rc = SQLITE_NOMEM;
            //  goto balance_cleanup;
            //}
            if (szCell.Length < nMaxCells)
                Array.Resize(ref szCell, nMaxCells); //(u16*)&apCell[nMaxCells];
            //aSpace1 = new byte[pBt.pageSize * (nMaxCells)];//  aSpace1 = (u8*)&szCell[nMaxCells];
            //Debug.Assert( EIGHT_BYTE_ALIGNMENT(aSpace1) );

            /*
            ** Load pointers to all cells on sibling pages and the divider cells
            ** into the local apCell[] array.  Make copies of the divider cells
            ** into space obtained from aSpace1[] and remove the the divider Cells
            ** from pParent.
            **
            ** If the siblings are on leaf pages, then the child pointers of the
            ** divider cells are stripped from the cells before they are copied
            ** into aSpace1[].  In this way, all cells in apCell[] are without
            ** child pointers.  If siblings are not leaves, then all cell in
            ** apCell[] include child pointers.  Either way, all cells in apCell[]
            ** are alike.
            **
            ** leafCorrection:  4 if pPage is a leaf.  0 if pPage is not a leaf.
            **       leafData:  1 if pPage holds key+data and pParent holds only keys.
            */
            leafCorrection = (u16)(apOld[0].leaf * 4);
            leafData = apOld[0].hasData;
            for (i = 0; i < nOld; i++)
            {
                int limit;

                /* Before doing anything else, take a copy of the i'th original sibling
                ** The rest of this function will use data from the copies rather
                ** that the original pages since the original pages will be in the
                ** process of being overwritten.  */
                //MemPage pOld = apCopy[i] = (MemPage*)&aSpace1[pBt.pageSize + k*i];
                //memcpy(pOld, apOld[i], sizeof(MemPage));
                //pOld.aData = (void*)&pOld[1];
                //memcpy(pOld.aData, apOld[i].aData, pBt.pageSize);
                MemPage pOld = apCopy[i] = apOld[i].Copy();

                limit = pOld.nCell + pOld.nOverflow;
                if (pOld.nOverflow > 0 || true)
                {
                    for (j = 0; j < limit; j++)
                    {
                        Debug.Assert(nCell < nMaxCells);
                        //apCell[nCell] = findOverflowCell( pOld, j );
                        //szCell[nCell] = cellSizePtr( pOld, apCell, nCell );
                        int iFOFC = findOverflowCell(pOld, j);
                        szCell[nCell] = cellSizePtr(pOld, iFOFC);
                        // Copy the Data Locally
                        if (apCell[nCell] == null)
                            apCell[nCell] = new u8[szCell[nCell]];
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
                    u8[] aData = pOld.aData;
                    u16 maskPage = pOld.maskPage;
                    u16 cellOffset = pOld.cellOffset;
                    for (j = 0; j < limit; j++)
                    {
                        Debugger.Break();
                        Debug.Assert(nCell < nMaxCells);
                        apCell[nCell] = findCellv2(aData, maskPage, cellOffset, j);
                        szCell[nCell] = cellSizePtr(pOld, apCell[nCell]);
                        nCell++;
                    }
                }
                if (i < nOld - 1 && 0 == leafData)
                {
                    u16 sz = (u16)szNew[i];
                    byte[] pTemp = sqlite3Malloc(sz + leafCorrection);
                    Debug.Assert(nCell < nMaxCells);
                    szCell[nCell] = sz;
                    //pTemp = &aSpace1[iSpace1];
                    //iSpace1 += sz;
                    Debug.Assert(sz <= pBt.maxLocal + 23);
                    //Debug.Assert(iSpace1 <= (int)pBt.pageSize);
                    Buffer.BlockCopy(pParent.aData, apDiv[i], pTemp, 0, sz);//memcpy( pTemp, apDiv[i], sz );
                    if (apCell[nCell] == null || apCell[nCell].Length < sz)
                        Array.Resize(ref apCell[nCell], sz);
                    Buffer.BlockCopy(pTemp, leafCorrection, apCell[nCell], 0, sz);//apCell[nCell] = pTemp + leafCorrection;
                    Debug.Assert(leafCorrection == 0 || leafCorrection == 4);
                    szCell[nCell] = (u16)(szCell[nCell] - leafCorrection);
                    if (0 == pOld.leaf)
                    {
                        Debug.Assert(leafCorrection == 0);
                        Debug.Assert(pOld.hdrOffset == 0);
                        /* The right pointer of the child page pOld becomes the left
                        ** pointer of the divider cell */
                        Buffer.BlockCopy(pOld.aData, 8, apCell[nCell], 0, 4);//memcpy( apCell[nCell], ref pOld.aData[8], 4 );
                    }
                    else
                    {
                        Debug.Assert(leafCorrection == 4);
                        if (szCell[nCell] < 4)
                        {
                            /* Do not allow any cells smaller than 4 bytes. */
                            szCell[nCell] = 4;
                        }
                    }
                    nCell++;
                }
            }

            /*
            ** Figure out the number of pages needed to hold all nCell cells.
            ** Store this number in "k".  Also compute szNew[] which is the total
            ** size of all cells on the i-th page and cntNew[] which is the index
            ** in apCell[] of the cell that divides page i from page i+1.
            ** cntNew[k] should equal nCell.
            **
            ** Values computed by this block:
            **
            **           k: The total number of sibling pages
            **    szNew[i]: Spaced used on the i-th sibling page.
            **   cntNew[i]: Index in apCell[] and szCell[] for the first cell to
            **              the right of the i-th sibling page.
            ** usableSpace: Number of bytes of space available on each sibling.
            **
            */
            usableSpace = (int)pBt.usableSize - 12 + leafCorrection;
            for (subtotal = k = i = 0; i < nCell; i++)
            {
                Debug.Assert(i < nMaxCells);
                subtotal += szCell[i] + 2;
                if (subtotal > usableSpace)
                {
                    szNew[k] = subtotal - szCell[i];
                    cntNew[k] = i;
                    if (leafData != 0)
                    {
                        i--;
                    }
                    subtotal = 0;
                    k++;
                    if (k > NB + 1)
                    {
                        rc = SQLITE_CORRUPT_BKPT();
                        goto balance_cleanup;
                    }
                }
            }
            szNew[k] = subtotal;
            cntNew[k] = nCell;
            k++;

            /*
            ** The packing computed by the previous block is biased toward the siblings
            ** on the left side.  The left siblings are always nearly full, while the
            ** right-most sibling might be nearly empty.  This block of code attempts
            ** to adjust the packing of siblings to get a better balance.
            **
            ** This adjustment is more than an optimization.  The packing above might
            ** be so out of balance as to be illegal.  For example, the right-most
            ** sibling might be completely empty.  This adjustment is not optional.
            */
            for (i = k - 1; i > 0; i--)
            {
                int szRight = szNew[i];  /* Size of sibling on the right */
                int szLeft = szNew[i - 1]; /* Size of sibling on the left */
                int r;              /* Index of right-most cell in left sibling */
                int d;              /* Index of first cell to the left of right sibling */

                r = cntNew[i - 1] - 1;
                d = r + 1 - leafData;
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

            /* Either we found one or more cells (cntnew[0])>0) or pPage is
            ** a virtual root page.  A virtual root page is when the real root
            ** page is page 1 and we are the only child of that page.
            */
            Debug.Assert(cntNew[0] > 0 || (pParent.pgno == 1 && pParent.nCell == 0));

            TRACE("BALANCE: old: %d %d %d  ",
            apOld[0].pgno,
            nOld >= 2 ? apOld[1].pgno : 0,
            nOld >= 3 ? apOld[2].pgno : 0
            );

            /*
            ** Allocate k new pages.  Reuse old pages where possible.
            */
            if (apOld[0].pgno <= 1)
            {
                rc = SQLITE_CORRUPT_BKPT();
                goto balance_cleanup;
            }
            pageFlags = apOld[0].aData[0];
            for (i = 0; i < k; i++)
            {
                MemPage pNew = new MemPage();
                if (i < nOld)
                {
                    pNew = apNew[i] = apOld[i];
                    apOld[i] = null;
                    rc = sqlite3PagerWrite(pNew.pDbPage);
                    nNew++;
                    if (rc != 0)
                        goto balance_cleanup;
                }
                else
                {
                    Debug.Assert(i > 0);
                    rc = allocateBtreePage(pBt, ref pNew, ref pgno, pgno, 0);
                    if (rc != 0)
                        goto balance_cleanup;
                    apNew[i] = pNew;
                    nNew++;

                    /* Set the pointer-map entry for the new sibling page. */
#if !SQLITE_OMIT_AUTOVACUUM //   if ( ISAUTOVACUUM )
                    if (pBt.autoVacuum)
#else
if (false)
#endif
                    {
                        ptrmapPut(pBt, pNew.pgno, PTRMAP_BTREE, pParent.pgno, ref rc);
                        if (rc != SQLITE_OK)
                        {
                            goto balance_cleanup;
                        }
                    }
                }
            }

            /* Free any old pages that were not reused as new pages.
            */
            while (i < nOld)
            {
                freePage(apOld[i], ref rc);
                if (rc != 0)
                    goto balance_cleanup;
                releasePage(apOld[i]);
                apOld[i] = null;
                i++;
            }

            /*
            ** Put the new pages in accending order.  This helps to
            ** keep entries in the disk file in order so that a scan
            ** of the table is a linear scan through the file.  That
            ** in turn helps the operating system to deliver pages
            ** from the disk more rapidly.
            **
            ** An O(n^2) insertion sort algorithm is used, but since
            ** n is never more than NB (a small constant), that should
            ** not be a problem.
            **
            ** When NB==3, this one optimization makes the database
            ** about 25% faster for large insertions and deletions.
            */
            for (i = 0; i < k - 1; i++)
            {
                int minV = (int)apNew[i].pgno;
                int minI = i;
                for (j = i + 1; j < k; j++)
                {
                    if (apNew[j].pgno < (u32)minV)
                    {
                        minI = j;
                        minV = (int)apNew[j].pgno;
                    }
                }
                if (minI > i)
                {
                    MemPage pT;
                    pT = apNew[i];
                    apNew[i] = apNew[minI];
                    apNew[minI] = pT;
                }
            }
            TRACE("new: %d(%d) %d(%d) %d(%d) %d(%d) %d(%d)\n",
            apNew[0].pgno, szNew[0],
            nNew >= 2 ? apNew[1].pgno : 0, nNew >= 2 ? szNew[1] : 0,
            nNew >= 3 ? apNew[2].pgno : 0, nNew >= 3 ? szNew[2] : 0,
            nNew >= 4 ? apNew[3].pgno : 0, nNew >= 4 ? szNew[3] : 0,
            nNew >= 5 ? apNew[4].pgno : 0, nNew >= 5 ? szNew[4] : 0);

            Debug.Assert(sqlite3PagerIswriteable(pParent.pDbPage));
            sqlite3Put4byte(pParent.aData, pRight, apNew[nNew - 1].pgno);

            /*
            ** Evenly distribute the data in apCell[] across the new pages.
            ** Insert divider cells into pParent as necessary.
            */
            j = 0;
            for (i = 0; i < nNew; i++)
            {
                /* Assemble the new sibling page. */
                MemPage pNew = apNew[i];
                Debug.Assert(j < nMaxCells);
                zeroPage(pNew, pageFlags);
                assemblePage(pNew, cntNew[i] - j, apCell, szCell, j);
                Debug.Assert(pNew.nCell > 0 || (nNew == 1 && cntNew[0] == 0));
                Debug.Assert(pNew.nOverflow == 0);

                j = cntNew[i];

                /* If the sibling page assembled above was not the right-most sibling,
                ** insert a divider cell into the parent page.
                */
                Debug.Assert(i < nNew - 1 || j == nCell);
                if (j < nCell)
                {
                    u8[] pCell;
                    u8[] pTemp;
                    int sz;

                    Debug.Assert(j < nMaxCells);
                    pCell = apCell[j];
                    sz = szCell[j] + leafCorrection;
                    pTemp = sqlite3Malloc(sz);//&aOvflSpace[iOvflSpace];
                    if (0 == pNew.leaf)
                    {
                        Buffer.BlockCopy(pCell, 0, pNew.aData, 8, 4);//memcpy( pNew.aData[8], pCell, 4 );
                    }
                    else if (leafData != 0)
                    {
                        /* If the tree is a leaf-data tree, and the siblings are leaves,
                        ** then there is no divider cell in apCell[]. Instead, the divider
                        ** cell consists of the integer key for the right-most cell of
                        ** the sibling-page assembled above only.
                        */
                        CellInfo info = new CellInfo();
                        j--;
                        btreeParseCellPtr(pNew, apCell[j], ref info);
                        pCell = pTemp;
                        sz = 4 + putVarint(pCell, 4, (u64)info.nKey);
                        pTemp = null;
                    }
                    else
                    {
                        //------------ pCell -= 4;
                        byte[] _pCell_4 = sqlite3Malloc(pCell.Length + 4);
                        Buffer.BlockCopy(pCell, 0, _pCell_4, 4, pCell.Length);
                        pCell = _pCell_4;
                        //
                        /* Obscure case for non-leaf-data trees: If the cell at pCell was
                        ** previously stored on a leaf node, and its reported size was 4
                        ** bytes, then it may actually be smaller than this
                        ** (see btreeParseCellPtr(), 4 bytes is the minimum size of
                        ** any cell). But it is important to pass the correct size to
                        ** insertCell(), so reparse the cell now.
                        **
                        ** Note that this can never happen in an SQLite data file, as all
                        ** cells are at least 4 bytes. It only happens in b-trees used
                        ** to evaluate "IN (SELECT ...)" and similar clauses.
                        */
                        if (szCell[j] == 4)
                        {
                            Debug.Assert(leafCorrection == 4);
                            sz = cellSizePtr(pParent, pCell);
                        }
                    }
                    iOvflSpace += sz;
                    Debug.Assert(sz <= pBt.maxLocal + 23);
                    Debug.Assert(iOvflSpace <= (int)pBt.pageSize);
                    insertCell(pParent, nxDiv, pCell, sz, pTemp, pNew.pgno, ref rc);
                    if (rc != SQLITE_OK)
                        goto balance_cleanup;
                    Debug.Assert(sqlite3PagerIswriteable(pParent.pDbPage));

                    j++;
                    nxDiv++;
                }
            }
            Debug.Assert(j == nCell);
            Debug.Assert(nOld > 0);
            Debug.Assert(nNew > 0);
            if ((pageFlags & PTF_LEAF) == 0)
            {
                Buffer.BlockCopy(apCopy[nOld - 1].aData, 8, apNew[nNew - 1].aData, 8, 4); //u8* zChild = &apCopy[nOld - 1].aData[8];
                //memcpy( apNew[nNew - 1].aData[8], zChild, 4 );
            }

            if (isRoot != 0 && pParent.nCell == 0 && pParent.hdrOffset <= apNew[0].nFree)
            {
                /* The root page of the b-tree now contains no cells. The only sibling
                ** page is the right-child of the parent. Copy the contents of the
                ** child page into the parent, decreasing the overall height of the
                ** b-tree structure by one. This is described as the "balance-shallower"
                ** sub-algorithm in some documentation.
                **
                ** If this is an auto-vacuum database, the call to copyNodeContent()
                ** sets all pointer-map entries corresponding to database image pages
                ** for which the pointer is stored within the content being copied.
                **
                ** The second Debug.Assert below verifies that the child page is defragmented
                ** (it must be, as it was just reconstructed using assemblePage()). This
                ** is important if the parent page happens to be page 1 of the database
                ** image.  */
                Debug.Assert(nNew == 1);
                Debug.Assert(apNew[0].nFree ==
                (get2byte(apNew[0].aData, 5) - apNew[0].cellOffset - apNew[0].nCell * 2)
                );
                copyNodeContent(apNew[0], pParent, ref rc);
                freePage(apNew[0], ref rc);
            }
            else
#if !SQLITE_OMIT_AUTOVACUUM //   if ( ISAUTOVACUUM )
                if (pBt.autoVacuum)
#else
if (false)
#endif
                {
                    /* Fix the pointer-map entries for all the cells that were shifted around.
                    ** There are several different types of pointer-map entries that need to
                    ** be dealt with by this routine. Some of these have been set already, but
                    ** many have not. The following is a summary:
                    **
                    **   1) The entries associated with new sibling pages that were not
                    **      siblings when this function was called. These have already
                    **      been set. We don't need to worry about old siblings that were
                    **      moved to the free-list - the freePage() code has taken care
                    **      of those.
                    **
                    **   2) The pointer-map entries associated with the first overflow
                    **      page in any overflow chains used by new divider cells. These
                    **      have also already been taken care of by the insertCell() code.
                    **
                    **   3) If the sibling pages are not leaves, then the child pages of
                    **      cells stored on the sibling pages may need to be updated.
                    **
                    **   4) If the sibling pages are not internal intkey nodes, then any
                    **      overflow pages used by these cells may need to be updated
                    **      (internal intkey nodes never contain pointers to overflow pages).
                    **
                    **   5) If the sibling pages are not leaves, then the pointer-map
                    **      entries for the right-child pages of each sibling may need
                    **      to be updated.
                    **
                    ** Cases 1 and 2 are dealt with above by other code. The next
                    ** block deals with cases 3 and 4 and the one after that, case 5. Since
                    ** setting a pointer map entry is a relatively expensive operation, this
                    ** code only sets pointer map entries for child or overflow pages that have
                    ** actually moved between pages.  */
                    MemPage pNew = apNew[0];
                    MemPage pOld = apCopy[0];
                    int nOverflow = pOld.nOverflow;
                    int iNextOld = pOld.nCell + nOverflow;
                    int iOverflow = (nOverflow != 0 ? pOld.aOvfl[0].idx : -1);
                    j = 0;                             /* Current 'old' sibling page */
                    k = 0;                             /* Current 'new' sibling page */
                    for (i = 0; i < nCell; i++)
                    {
                        int isDivider = 0;
                        while (i == iNextOld)
                        {
                            /* Cell i is the cell immediately following the last cell on old
                            ** sibling page j. If the siblings are not leaf pages of an
                            ** intkey b-tree, then cell i was a divider cell. */
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
                            if ((--nOverflow) > 0)
                            {
                                iOverflow++;
                            }
                        }

                        if (i == cntNew[k])
                        {
                            /* Cell i is the cell immediately following the last cell on new
                            ** sibling page k. If the siblings are not leaf pages of an
                            ** intkey b-tree, then cell i is a divider cell.  */
                            pNew = apNew[++k];
                            if (0 == leafData)
                                continue;
                        }
                        Debug.Assert(j < nOld);
                        Debug.Assert(k < nNew);

                        /* If the cell was originally divider cell (and is not now) or
                        ** an overflow cell, or if the cell was located on a different sibling
                        ** page before the balancing, then the pointer map entries associated
                        ** with any child or overflow pages need to be updated.  */
                        if (isDivider != 0 || pOld.pgno != pNew.pgno)
                        {
                            if (0 == leafCorrection)
                            {
                                ptrmapPut(pBt, sqlite3Get4byte(apCell[i]), PTRMAP_BTREE, pNew.pgno, ref rc);
                            }
                            if (szCell[i] > pNew.minLocal)
                            {
                                ptrmapPutOvflPtr(pNew, apCell[i], ref rc);
                            }
                        }
                    }

                    if (0 == leafCorrection)
                    {
                        for (i = 0; i < nNew; i++)
                        {
                            u32 key = sqlite3Get4byte(apNew[i].aData, 8);
                            ptrmapPut(pBt, key, PTRMAP_BTREE, apNew[i].pgno, ref rc);
                        }
                    }

#if FALSE
/* The ptrmapCheckPages() contains Debug.Assert() statements that verify that
** all pointer map pages are set correctly. This is helpful while
** debugging. This is usually disabled because a corrupt database may
** cause an Debug.Assert() statement to fail.  */
ptrmapCheckPages(apNew, nNew);
ptrmapCheckPages(pParent, 1);
#endif
                }

            Debug.Assert(pParent.isInit != 0);
            TRACE("BALANCE: finished: old=%d new=%d cells=%d\n",
            nOld, nNew, nCell);

        /*
        ** Cleanup before returning.
        */
        balance_cleanup:
            sqlite3ScratchFree(apCell);
            for (i = 0; i < nOld; i++)
            {
                releasePage(apOld[i]);
            }
            for (i = 0; i < nNew; i++)
            {
                releasePage(apNew[i]);
            }

            return rc;
        }


        /*
        ** This function is called when the root page of a b-tree structure is
        ** overfull (has one or more overflow pages).
        **
        ** A new child page is allocated and the contents of the current root
        ** page, including overflow cells, are copied into the child. The root
        ** page is then overwritten to make it an empty page with the right-child
        ** pointer pointing to the new page.
        **
        ** Before returning, all pointer-map entries corresponding to pages
        ** that the new child-page now contains pointers to are updated. The
        ** entry corresponding to the new right-child pointer of the root
        ** page is also updated.
        **
        ** If successful, ppChild is set to contain a reference to the child
        ** page and SQLITE_OK is returned. In this case the caller is required
        ** to call releasePage() on ppChild exactly once. If an error occurs,
        ** an error code is returned and ppChild is set to 0.
        */
        static int balance_deeper(MemPage pRoot, ref MemPage ppChild)
        {
            int rc;                        /* Return value from subprocedures */
            MemPage pChild = null;           /* Pointer to a new child page */
            Pgno pgnoChild = 0;            /* Page number of the new child page */
            BtShared pBt = pRoot.pBt;    /* The BTree */

            Debug.Assert(pRoot.nOverflow > 0);
            Debug.Assert(sqlite3_mutex_held(pBt.mutex));

            /* Make pRoot, the root page of the b-tree, writable. Allocate a new
            ** page that will become the new right-child of pPage. Copy the contents
            ** of the node stored on pRoot into the new child page.
            */
            rc = sqlite3PagerWrite(pRoot.pDbPage);
            if (rc == SQLITE_OK)
            {
                rc = allocateBtreePage(pBt, ref pChild, ref pgnoChild, pRoot.pgno, 0);
                copyNodeContent(pRoot, pChild, ref rc);
#if !SQLITE_OMIT_AUTOVACUUM //   if ( ISAUTOVACUUM )
                if (pBt.autoVacuum)
#else
if (false)
#endif
                {
                    ptrmapPut(pBt, pgnoChild, PTRMAP_BTREE, pRoot.pgno, ref rc);
                }
            }
            if (rc != 0)
            {
                ppChild = null;
                releasePage(pChild);
                return rc;
            }
            Debug.Assert(sqlite3PagerIswriteable(pChild.pDbPage));
            Debug.Assert(sqlite3PagerIswriteable(pRoot.pDbPage));
            Debug.Assert(pChild.nCell == pRoot.nCell);

            TRACE("BALANCE: copy root %d into %d\n", pRoot.pgno, pChild.pgno);

            /* Copy the overflow cells from pRoot to pChild */
            Array.Copy(pRoot.aOvfl, pChild.aOvfl, pRoot.nOverflow);//memcpy(pChild.aOvfl, pRoot.aOvfl, pRoot.nOverflow*sizeof(pRoot.aOvfl[0]));
            pChild.nOverflow = pRoot.nOverflow;

            /* Zero the contents of pRoot. Then install pChild as the right-child. */
            zeroPage(pRoot, pChild.aData[0] & ~PTF_LEAF);
            sqlite3Put4byte(pRoot.aData, pRoot.hdrOffset + 8, pgnoChild);

            ppChild = pChild;
            return SQLITE_OK;
        }

        /*
        ** The page that pCur currently points to has just been modified in
        ** some way. This function figures out if this modification means the
        ** tree needs to be balanced, and if so calls the appropriate balancing
        ** routine. Balancing routines are:
        **
        **   balance_quick()
        **   balance_deeper()
        **   balance_nonroot()
        */
        static u8[] aBalanceQuickSpace = new u8[13];
        static int balance(BtCursor pCur)
        {
            int rc = SQLITE_OK;
            int nMin = (int)pCur.pBt.usableSize * 2 / 3;

            //u8[] pFree = null;

#if !NDEBUG || SQLITE_COVERAGE_TEST || DEBUG
            int balance_quick_called = 0;//TESTONLY( int balance_quick_called = 0 );
            int balance_deeper_called = 0;//TESTONLY( int balance_deeper_called = 0 );
#else
int balance_quick_called = 0;
int balance_deeper_called = 0;
#endif

            do
            {
                int iPage = pCur.iPage;
                MemPage pPage = pCur.apPage[iPage];

                if (iPage == 0)
                {
                    if (pPage.nOverflow != 0)
                    {
                        /* The root page of the b-tree is overfull. In this case call the
                        ** balance_deeper() function to create a new child for the root-page
                        ** and copy the current contents of the root-page to it. The
                        ** next iteration of the do-loop will balance the child page.
                        */
                        Debug.Assert((balance_deeper_called++) == 0);
                        rc = balance_deeper(pPage, ref pCur.apPage[1]);
                        if (rc == SQLITE_OK)
                        {
                            pCur.iPage = 1;
                            pCur.aiIdx[0] = 0;
                            pCur.aiIdx[1] = 0;
                            Debug.Assert(pCur.apPage[1].nOverflow != 0);
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                else if (pPage.nOverflow == 0 && pPage.nFree <= nMin)
                {
                    break;
                }
                else
                {
                    MemPage pParent = pCur.apPage[iPage - 1];
                    int iIdx = pCur.aiIdx[iPage - 1];

                    rc = sqlite3PagerWrite(pParent.pDbPage);
                    if (rc == SQLITE_OK)
                    {
#if !SQLITE_OMIT_QUICKBALANCE
                        if (pPage.hasData != 0
                        && pPage.nOverflow == 1
                        && pPage.aOvfl[0].idx == pPage.nCell
                        && pParent.pgno != 1
                        && pParent.nCell == iIdx
                        )
                        {
                            /* Call balance_quick() to create a new sibling of pPage on which
                            ** to store the overflow cell. balance_quick() inserts a new cell
                            ** into pParent, which may cause pParent overflow. If this
                            ** happens, the next interation of the do-loop will balance pParent
                            ** use either balance_nonroot() or balance_deeper(). Until this
                            ** happens, the overflow cell is stored in the aBalanceQuickSpace[]
                            ** buffer.
                            **
                            ** The purpose of the following Debug.Assert() is to check that only a
                            ** single call to balance_quick() is made for each call to this
                            ** function. If this were not verified, a subtle bug involving reuse
                            ** of the aBalanceQuickSpace[] might sneak in.
                            */
                            Debug.Assert((balance_quick_called++) == 0);
                            rc = balance_quick(pParent, pPage, aBalanceQuickSpace);
                        }
                        else
#endif
                        {
                            /* In this case, call balance_nonroot() to redistribute cells
                            ** between pPage and up to 2 of its sibling pages. This involves
                            ** modifying the contents of pParent, which may cause pParent to
                            ** become overfull or underfull. The next iteration of the do-loop
                            ** will balance the parent page to correct this.
                            **
                            ** If the parent page becomes overfull, the overflow cell or cells
                            ** are stored in the pSpace buffer allocated immediately below.
                            ** A subsequent iteration of the do-loop will deal with this by
                            ** calling balance_nonroot() (balance_deeper() may be called first,
                            ** but it doesn't deal with overflow cells - just moves them to a
                            ** different page). Once this subsequent call to balance_nonroot()
                            ** has completed, it is safe to release the pSpace buffer used by
                            ** the previous call, as the overflow cell data will have been
                            ** copied either into the body of a database page or into the new
                            ** pSpace buffer passed to the latter call to balance_nonroot().
                            */
                            u8[] pSpace = new u8[pCur.pBt.pageSize];// u8 pSpace = sqlite3PageMalloc( pCur.pBt.pageSize );
                            rc = balance_nonroot(pParent, iIdx, null, iPage == 1 ? 1 : 0);
                            //if (pFree != null)
                            //{
                            //  /* If pFree is not NULL, it points to the pSpace buffer used
                            //  ** by a previous call to balance_nonroot(). Its contents are
                            //  ** now stored either on real database pages or within the
                            //  ** new pSpace buffer, so it may be safely freed here. */
                            //  sqlite3PageFree(ref pFree);
                            //}

                            /* The pSpace buffer will be freed after the next call to
                            ** balance_nonroot(), or just before this function returns, whichever
                            ** comes first. */
                            //pFree = pSpace;
                        }
                    }

                    pPage.nOverflow = 0;

                    /* The next iteration of the do-loop balances the parent page. */
                    releasePage(pPage);
                    pCur.iPage--;
                }
            } while (rc == SQLITE_OK);

            //if (pFree != null)
            //{
            //  sqlite3PageFree(ref pFree);
            //}
            return rc;
        }


        /*
        ** Insert a new record into the BTree.  The key is given by (pKey,nKey)
        ** and the data is given by (pData,nData).  The cursor is used only to
        ** define what table the record should be inserted into.  The cursor
        ** is left pointing at a random location.
        **
        ** For an INTKEY table, only the nKey value of the key is used.  pKey is
        ** ignored.  For a ZERODATA table, the pData and nData are both ignored.
        **
        ** If the seekResult parameter is non-zero, then a successful call to
        ** MovetoUnpacked() to seek cursor pCur to (pKey, nKey) has already
        ** been performed. seekResult is the search result returned (a negative
        ** number if pCur points at an entry that is smaller than (pKey, nKey), or
        ** a positive value if pCur points at an etry that is larger than
        ** (pKey, nKey)).
        **
        ** If the seekResult parameter is non-zero, then the caller guarantees that
        ** cursor pCur is pointing at the existing copy of a row that is to be
        ** overwritten.  If the seekResult parameter is 0, then cursor pCur may
        ** point to any entry or to no entry at all and so this function has to seek
        ** the cursor before the new key can be inserted.
        */
        static int sqlite3BtreeInsert(
        BtCursor pCur,                /* Insert data into the table of this cursor */
        byte[] pKey, i64 nKey,        /* The key of the new record */
        byte[] pData, int nData,      /* The data of the new record */
        int nZero,                    /* Number of extra 0 bytes to append to data */
        int appendBias,               /* True if this is likely an append */
        int seekResult                /* Result of prior MovetoUnpacked() call */
        )
        {
            int rc;
            int loc = seekResult;       /* -1: before desired location  +1: after */
            int szNew = 0;
            int idx;
            MemPage pPage;
            Btree p = pCur.pBtree;
            BtShared pBt = p.pBt;
            int oldCell;
            byte[] newCell = null;

            if (pCur.eState == CURSOR_FAULT)
            {
                Debug.Assert(pCur.skipNext != SQLITE_OK);
                return pCur.skipNext;
            }

            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.wrFlag != 0 && pBt.inTransaction == TRANS_WRITE && !pBt.readOnly);
            Debug.Assert(hasSharedCacheTableLock(p, pCur.pgnoRoot, pCur.pKeyInfo != null ? 1 : 0, 2));

            /* Assert that the caller has been consistent. If this cursor was opened
            ** expecting an index b-tree, then the caller should be inserting blob
            ** keys with no associated data. If the cursor was opened expecting an
            ** intkey table, the caller should be inserting integer keys with a
            ** blob of associated data.  */
            Debug.Assert((pKey == null) == (pCur.pKeyInfo == null));

            /* If this is an insert into a table b-tree, invalidate any incrblob
            ** cursors open on the row being replaced (assuming this is a replace
            ** operation - if it is not, the following is a no-op).  */
            if (pCur.pKeyInfo == null)
            {
                invalidateIncrblobCursors(p, nKey, 0);
            }

            /* Save the positions of any other cursors open on this table.
            **
            ** In some cases, the call to btreeMoveto() below is a no-op. For
            ** example, when inserting data into a table with auto-generated integer
            ** keys, the VDBE layer invokes sqlite3BtreeLast() to figure out the
            ** integer key to use. It then calls this function to actually insert the
            ** data into the intkey B-Tree. In this case btreeMoveto() recognizes
            ** that the cursor is already where it needs to be and returns without
            ** doing any work. To avoid thwarting these optimizations, it is important
            ** not to clear the cursor here.
            */
            rc = saveAllCursors(pBt, pCur.pgnoRoot, pCur);
            if (rc != 0)
                return rc;
            if (0 == loc)
            {
                rc = btreeMoveto(pCur, pKey, nKey, appendBias, ref loc);
                if (rc != 0)
                    return rc;
            }
            Debug.Assert(pCur.eState == CURSOR_VALID || (pCur.eState == CURSOR_INVALID && loc != 0));

            pPage = pCur.apPage[pCur.iPage];
            Debug.Assert(pPage.intKey != 0 || nKey >= 0);
            Debug.Assert(pPage.leaf != 0 || 0 == pPage.intKey);

            TRACE("INSERT: table=%d nkey=%lld ndata=%d page=%d %s\n",
            pCur.pgnoRoot, nKey, nData, pPage.pgno,
            loc == 0 ? "overwrite" : "new entry");
            Debug.Assert(pPage.isInit != 0);
            allocateTempSpace(pBt);
            newCell = pBt.pTmpSpace;
            //if (newCell == null) return SQLITE_NOMEM;
            rc = fillInCell(pPage, newCell, pKey, nKey, pData, nData, nZero, ref szNew);
            if (rc != 0)
                goto end_insert;
            Debug.Assert(szNew == cellSizePtr(pPage, newCell));
            Debug.Assert(szNew <= MX_CELL_SIZE(pBt));
            idx = pCur.aiIdx[pCur.iPage];
            if (loc == 0)
            {
                u16 szOld;
                Debug.Assert(idx < pPage.nCell);
                rc = sqlite3PagerWrite(pPage.pDbPage);
                if (rc != 0)
                {
                    goto end_insert;
                }
                oldCell = findCell(pPage, idx);
                if (0 == pPage.leaf)
                {
                    //memcpy(newCell, oldCell, 4);
                    newCell[0] = pPage.aData[oldCell + 0];
                    newCell[1] = pPage.aData[oldCell + 1];
                    newCell[2] = pPage.aData[oldCell + 2];
                    newCell[3] = pPage.aData[oldCell + 3];
                }
                szOld = cellSizePtr(pPage, oldCell);
                rc = clearCell(pPage, oldCell);
                dropCell(pPage, idx, szOld, ref rc);
                if (rc != 0)
                    goto end_insert;
            }
            else if (loc < 0 && pPage.nCell > 0)
            {
                Debug.Assert(pPage.leaf != 0);
                idx = ++pCur.aiIdx[pCur.iPage];
            }
            else
            {
                Debug.Assert(pPage.leaf != 0);
            }
            insertCell(pPage, idx, newCell, szNew, null, 0, ref rc);
            Debug.Assert(rc != SQLITE_OK || pPage.nCell > 0 || pPage.nOverflow > 0);

            /* If no error has occured and pPage has an overflow cell, call balance()
            ** to redistribute the cells within the tree. Since balance() may move
            ** the cursor, zero the BtCursor.info.nSize and BtCursor.validNKey
            ** variables.
            **
            ** Previous versions of SQLite called moveToRoot() to move the cursor
            ** back to the root page as balance() used to invalidate the contents
            ** of BtCursor.apPage[] and BtCursor.aiIdx[]. Instead of doing that,
            ** set the cursor state to "invalid". This makes common insert operations
            ** slightly faster.
            **
            ** There is a subtle but important optimization here too. When inserting
            ** multiple records into an intkey b-tree using a single cursor (as can
            ** happen while processing an "INSERT INTO ... SELECT" statement), it
            ** is advantageous to leave the cursor pointing to the last entry in
            ** the b-tree if possible. If the cursor is left pointing to the last
            ** entry in the table, and the next row inserted has an integer key
            ** larger than the largest existing key, it is possible to insert the
            ** row without seeking the cursor. This can be a big performance boost.
            */
            pCur.info.nSize = 0;
            pCur.validNKey = false;
            if (rc == SQLITE_OK && pPage.nOverflow != 0)
            {
                rc = balance(pCur);

                /* Must make sure nOverflow is reset to zero even if the balance()
                ** fails. Internal data structure corruption will result otherwise.
                ** Also, set the cursor state to invalid. This stops saveCursorPosition()
                ** from trying to save the current position of the cursor.  */
                pCur.apPage[pCur.iPage].nOverflow = 0;
                pCur.eState = CURSOR_INVALID;
            }
            Debug.Assert(pCur.apPage[pCur.iPage].nOverflow == 0);

        end_insert:
            return rc;
        }

        /*
        ** Delete the entry that the cursor is pointing to.  The cursor
        ** is left pointing at a arbitrary location.
        */
        static int sqlite3BtreeDelete(BtCursor pCur)
        {
            Btree p = pCur.pBtree;
            BtShared pBt = p.pBt;
            int rc;                             /* Return code */
            MemPage pPage;                      /* Page to delete cell from */
            int pCell;                          /* Pointer to cell to delete */
            int iCellIdx;                       /* Index of cell to delete */
            int iCellDepth;                     /* Depth of node containing pCell */

            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pBt.inTransaction == TRANS_WRITE);
            Debug.Assert(!pBt.readOnly);
            Debug.Assert(pCur.wrFlag != 0);
            Debug.Assert(hasSharedCacheTableLock(p, pCur.pgnoRoot, pCur.pKeyInfo != null ? 1 : 0, 2));
            Debug.Assert(!hasReadConflicts(p, pCur.pgnoRoot));

            if (NEVER(pCur.aiIdx[pCur.iPage] >= pCur.apPage[pCur.iPage].nCell)
            || NEVER(pCur.eState != CURSOR_VALID)
            )
            {
                return SQLITE_ERROR;  /* Something has gone awry. */
            }

            /* If this is a delete operation to remove a row from a table b-tree,
            ** invalidate any incrblob cursors open on the row being deleted.  */
            if (pCur.pKeyInfo == null)
            {
                invalidateIncrblobCursors(p, pCur.info.nKey, 0);
            }

            iCellDepth = pCur.iPage;
            iCellIdx = pCur.aiIdx[iCellDepth];
            pPage = pCur.apPage[iCellDepth];
            pCell = findCell(pPage, iCellIdx);

            /* If the page containing the entry to delete is not a leaf page, move
            ** the cursor to the largest entry in the tree that is smaller than
            ** the entry being deleted. This cell will replace the cell being deleted
            ** from the internal node. The 'previous' entry is used for this instead
            ** of the 'next' entry, as the previous entry is always a part of the
            ** sub-tree headed by the child page of the cell being deleted. This makes
            ** balancing the tree following the delete operation easier.  */
            if (0 == pPage.leaf)
            {
                int notUsed = 0;
                rc = sqlite3BtreePrevious(pCur, ref notUsed);
                if (rc != 0)
                    return rc;
            }

            /* Save the positions of any other cursors open on this table before
            ** making any modifications. Make the page containing the entry to be
            ** deleted writable. Then free any overflow pages associated with the
            ** entry and finally remove the cell itself from within the page.
            */
            rc = saveAllCursors(pBt, pCur.pgnoRoot, pCur);
            if (rc != 0)
                return rc;
            rc = sqlite3PagerWrite(pPage.pDbPage);
            if (rc != 0)
                return rc;
            rc = clearCell(pPage, pCell);
            dropCell(pPage, iCellIdx, cellSizePtr(pPage, pCell), ref rc);
            if (rc != 0)
                return rc;

            /* If the cell deleted was not located on a leaf page, then the cursor
            ** is currently pointing to the largest entry in the sub-tree headed
            ** by the child-page of the cell that was just deleted from an internal
            ** node. The cell from the leaf node needs to be moved to the internal
            ** node to replace the deleted cell.  */
            if (0 == pPage.leaf)
            {
                MemPage pLeaf = pCur.apPage[pCur.iPage];
                int nCell;
                Pgno n = pCur.apPage[iCellDepth + 1].pgno;
                //byte[] pTmp;

                pCell = findCell(pLeaf, pLeaf.nCell - 1);
                nCell = cellSizePtr(pLeaf, pCell);
                Debug.Assert(MX_CELL_SIZE(pBt) >= nCell);

                //allocateTempSpace(pBt);
                //pTmp = pBt.pTmpSpace;

                rc = sqlite3PagerWrite(pLeaf.pDbPage);
                byte[] pNext_4 = sqlite3Malloc(nCell + 4);
                Buffer.BlockCopy(pLeaf.aData, pCell - 4, pNext_4, 0, nCell + 4);
                insertCell(pPage, iCellIdx, pNext_4, nCell + 4, null, n, ref rc); //insertCell( pPage, iCellIdx, pCell - 4, nCell + 4, pTmp, n, ref rc );
                dropCell(pLeaf, pLeaf.nCell - 1, nCell, ref rc);
                if (rc != 0)
                    return rc;
            }

            /* Balance the tree. If the entry deleted was located on a leaf page,
            ** then the cursor still points to that page. In this case the first
            ** call to balance() repairs the tree, and the if(...) condition is
            ** never true.
            **
            ** Otherwise, if the entry deleted was on an internal node page, then
            ** pCur is pointing to the leaf page from which a cell was removed to
            ** replace the cell deleted from the internal node. This is slightly
            ** tricky as the leaf node may be underfull, and the internal node may
            ** be either under or overfull. In this case run the balancing algorithm
            ** on the leaf node first. If the balance proceeds far enough up the
            ** tree that we can be sure that any problem in the internal node has
            ** been corrected, so be it. Otherwise, after balancing the leaf node,
            ** walk the cursor up the tree to the internal node and balance it as
            ** well.  */
            rc = balance(pCur);
            if (rc == SQLITE_OK && pCur.iPage > iCellDepth)
            {
                while (pCur.iPage > iCellDepth)
                {
                    releasePage(pCur.apPage[pCur.iPage--]);
                }
                rc = balance(pCur);
            }

            if (rc == SQLITE_OK)
            {
                moveToRoot(pCur);
            }
            return rc;
        }

        /*
        ** Create a new BTree table.  Write into piTable the page
        ** number for the root page of the new table.
        **
        ** The type of type is determined by the flags parameter.  Only the
        ** following values of flags are currently in use.  Other values for
        ** flags might not work:
        **
        **     BTREE_INTKEY|BTREE_LEAFDATA     Used for SQL tables with rowid keys
        **     BTREE_ZERODATA                  Used for SQL indices
        */
        static int btreeCreateTable(Btree p, ref int piTable, int createTabFlags)
        {
            BtShared pBt = p.pBt;
            MemPage pRoot = new MemPage();
            Pgno pgnoRoot = 0;
            int rc;
            int ptfFlags;          /* Page-type flage for the root page of new table */

            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            Debug.Assert(pBt.inTransaction == TRANS_WRITE);
            Debug.Assert(!pBt.readOnly);

#if SQLITE_OMIT_AUTOVACUUM
rc = allocateBtreePage(pBt, ref pRoot, ref pgnoRoot, 1, 0);
if( rc !=0){
return rc;
}
#else
            if (pBt.autoVacuum)
            {
                Pgno pgnoMove = 0;                    /* Move a page here to make room for the root-page */
                MemPage pPageMove = new MemPage();  /* The page to move to. */

                /* Creating a new table may probably require moving an existing database
                ** to make room for the new tables root page. In case this page turns
                ** out to be an overflow page, delete all overflow page-map caches
                ** held by open cursors.
                */
                invalidateAllOverflowCache(pBt);

                /* Read the value of meta[3] from the database to determine where the
                ** root page of the new table should go. meta[3] is the largest root-page
                ** created so far, so the new root-page is (meta[3]+1).
                */
                sqlite3BtreeGetMeta(p, BTREE_LARGEST_ROOT_PAGE, ref pgnoRoot);
                pgnoRoot++;

                /* The new root-page may not be allocated on a pointer-map page, or the
                ** PENDING_BYTE page.
                */
                while (pgnoRoot == PTRMAP_PAGENO(pBt, pgnoRoot) ||
                pgnoRoot == PENDING_BYTE_PAGE(pBt))
                {
                    pgnoRoot++;
                }
                Debug.Assert(pgnoRoot >= 3);

                /* Allocate a page. The page that currently resides at pgnoRoot will
                ** be moved to the allocated page (unless the allocated page happens
                ** to reside at pgnoRoot).
                */
                rc = allocateBtreePage(pBt, ref pPageMove, ref pgnoMove, pgnoRoot, 1);
                if (rc != SQLITE_OK)
                {
                    return rc;
                }

                if (pgnoMove != pgnoRoot)
                {
                    /* pgnoRoot is the page that will be used for the root-page of
                    ** the new table (assuming an error did not occur). But we were
                    ** allocated pgnoMove. If required (i.e. if it was not allocated
                    ** by extending the file), the current page at position pgnoMove
                    ** is already journaled.
                    */
                    u8 eType = 0;
                    Pgno iPtrPage = 0;

                    releasePage(pPageMove);

                    /* Move the page currently at pgnoRoot to pgnoMove. */
                    rc = btreeGetPage(pBt, pgnoRoot, ref pRoot, 0);
                    if (rc != SQLITE_OK)
                    {
                        return rc;
                    }
                    rc = ptrmapGet(pBt, pgnoRoot, ref eType, ref iPtrPage);
                    if (eType == PTRMAP_ROOTPAGE || eType == PTRMAP_FREEPAGE)
                    {
                        rc = SQLITE_CORRUPT_BKPT();
                    }
                    if (rc != SQLITE_OK)
                    {
                        releasePage(pRoot);
                        return rc;
                    }
                    Debug.Assert(eType != PTRMAP_ROOTPAGE);
                    Debug.Assert(eType != PTRMAP_FREEPAGE);
                    rc = relocatePage(pBt, pRoot, eType, iPtrPage, pgnoMove, 0);
                    releasePage(pRoot);

                    /* Obtain the page at pgnoRoot */
                    if (rc != SQLITE_OK)
                    {
                        return rc;
                    }
                    rc = btreeGetPage(pBt, pgnoRoot, ref pRoot, 0);
                    if (rc != SQLITE_OK)
                    {
                        return rc;
                    }
                    rc = sqlite3PagerWrite(pRoot.pDbPage);
                    if (rc != SQLITE_OK)
                    {
                        releasePage(pRoot);
                        return rc;
                    }
                }
                else
                {
                    pRoot = pPageMove;
                }

                /* Update the pointer-map and meta-data with the new root-page number. */
                ptrmapPut(pBt, pgnoRoot, PTRMAP_ROOTPAGE, 0, ref rc);
                if (rc != 0)
                {
                    releasePage(pRoot);
                    return rc;
                }

                /* When the new root page was allocated, page 1 was made writable in
                ** order either to increase the database filesize, or to decrement the
                ** freelist count.  Hence, the sqlite3BtreeUpdateMeta() call cannot fail.
                */
                Debug.Assert(sqlite3PagerIswriteable(pBt.pPage1.pDbPage));
                rc = sqlite3BtreeUpdateMeta(p, 4, pgnoRoot);
                if (NEVER(rc != 0))
                {
                    releasePage(pRoot);
                    return rc;
                }

            }
            else
            {
                rc = allocateBtreePage(pBt, ref pRoot, ref pgnoRoot, 1, 0);
                if (rc != 0)
                    return rc;
            }
#endif
            Debug.Assert(sqlite3PagerIswriteable(pRoot.pDbPage));
            if ((createTabFlags & BTREE_INTKEY) != 0)
            {
                ptfFlags = PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF;
            }
            else
            {
                ptfFlags = PTF_ZERODATA | PTF_LEAF;
            }
            zeroPage(pRoot, ptfFlags);
            sqlite3PagerUnref(pRoot.pDbPage);
            Debug.Assert((pBt.openFlags & BTREE_SINGLE) == 0 || pgnoRoot == 2);
            piTable = (int)pgnoRoot;
            return SQLITE_OK;
        }
        static int sqlite3BtreeCreateTable(Btree p, ref int piTable, int flags)
        {
            int rc;
            sqlite3BtreeEnter(p);
            rc = btreeCreateTable(p, ref piTable, flags);
            sqlite3BtreeLeave(p);
            return rc;
        }

        /*
        ** Erase the given database page and all its children.  Return
        ** the page to the freelist.
        */
        static int clearDatabasePage(
        BtShared pBt,         /* The BTree that contains the table */
        Pgno pgno,            /* Page number to clear */
        int freePageFlag,     /* Deallocate page if true */
        ref int pnChange      /* Add number of Cells freed to this counter */
        )
        {
            MemPage pPage = new MemPage();
            int rc;
            byte[] pCell;
            int i;

            Debug.Assert(sqlite3_mutex_held(pBt.mutex));
            if (pgno > btreePagecount(pBt))
            {
                return SQLITE_CORRUPT_BKPT();
            }

            rc = getAndInitPage(pBt, pgno, ref pPage);
            if (rc != 0)
                return rc;
            for (i = 0; i < pPage.nCell; i++)
            {
                int iCell = findCell(pPage, i);
                pCell = pPage.aData; //        pCell = findCell( pPage, i );
                if (0 == pPage.leaf)
                {
                    rc = clearDatabasePage(pBt, sqlite3Get4byte(pCell, iCell), 1, ref pnChange);
                    if (rc != 0)
                        goto cleardatabasepage_out;
                }
                rc = clearCell(pPage, iCell);
                if (rc != 0)
                    goto cleardatabasepage_out;
            }
            if (0 == pPage.leaf)
            {
                rc = clearDatabasePage(pBt, sqlite3Get4byte(pPage.aData, 8), 1, ref pnChange);
                if (rc != 0)
                    goto cleardatabasepage_out;
            }
            else //if (pnChange != 0)
            {
                //Debug.Assert(pPage.intKey != 0);
                pnChange += pPage.nCell;
            }
            if (freePageFlag != 0)
            {
                freePage(pPage, ref rc);
            }
            else if ((rc = sqlite3PagerWrite(pPage.pDbPage)) == 0)
            {
                zeroPage(pPage, pPage.aData[0] | PTF_LEAF);
            }

        cleardatabasepage_out:
            releasePage(pPage);
            return rc;
        }

        /*
        ** Delete all information from a single table in the database.  iTable is
        ** the page number of the root of the table.  After this routine returns,
        ** the root page is empty, but still exists.
        **
        ** This routine will fail with SQLITE_LOCKED if there are any open
        ** read cursors on the table.  Open write cursors are moved to the
        ** root of the table.
        **
        ** If pnChange is not NULL, then table iTable must be an intkey table. The
        ** integer value pointed to by pnChange is incremented by the number of
        ** entries in the table.
        */
        static int sqlite3BtreeClearTable(Btree p, int iTable, ref int pnChange)
        {
            int rc;
            BtShared pBt = p.pBt;
            sqlite3BtreeEnter(p);
            Debug.Assert(p.inTrans == TRANS_WRITE);

            /* Invalidate all incrblob cursors open on table iTable (assuming iTable
            ** is the root of a table b-tree - if it is not, the following call is
            ** a no-op).  */
            invalidateIncrblobCursors(p, 0, 1);

            rc = saveAllCursors(pBt, (Pgno)iTable, null);
            if (SQLITE_OK == rc)
            {
                rc = clearDatabasePage(pBt, (Pgno)iTable, 0, ref pnChange);
            }
            sqlite3BtreeLeave(p);
            return rc;
        }

        /*
        ** Erase all information in a table and add the root of the table to
        ** the freelist.  Except, the root of the principle table (the one on
        ** page 1) is never added to the freelist.
        **
        ** This routine will fail with SQLITE_LOCKED if there are any open
        ** cursors on the table.
        **
        ** If AUTOVACUUM is enabled and the page at iTable is not the last
        ** root page in the database file, then the last root page
        ** in the database file is moved into the slot formerly occupied by
        ** iTable and that last slot formerly occupied by the last root page
        ** is added to the freelist instead of iTable.  In this say, all
        ** root pages are kept at the beginning of the database file, which
        ** is necessary for AUTOVACUUM to work right.  piMoved is set to the
        ** page number that used to be the last root page in the file before
        ** the move.  If no page gets moved, piMoved is set to 0.
        ** The last root page is recorded in meta[3] and the value of
        ** meta[3] is updated by this procedure.
        */
        static int btreeDropTable(Btree p, Pgno iTable, ref int piMoved)
        {
            int rc;
            MemPage pPage = null;
            BtShared pBt = p.pBt;

            Debug.Assert(sqlite3BtreeHoldsMutex(p));
            Debug.Assert(p.inTrans == TRANS_WRITE);

            /* It is illegal to drop a table if any cursors are open on the
            ** database. This is because in auto-vacuum mode the backend may
            ** need to move another root-page to fill a gap left by the deleted
            ** root page. If an open cursor was using this page a problem would
            ** occur.
            **
            ** This error is caught long before control reaches this point.
            */
            if (NEVER(pBt.pCursor))
            {
                sqlite3ConnectionBlocked(p.db, pBt.pCursor.pBtree.db);
                return SQLITE_LOCKED_SHAREDCACHE;
            }

            rc = btreeGetPage(pBt, (Pgno)iTable, ref pPage, 0);
            if (rc != 0)
                return rc;
            int Dummy0 = 0;
            rc = sqlite3BtreeClearTable(p, (int)iTable, ref Dummy0);
            if (rc != 0)
            {
                releasePage(pPage);
                return rc;
            }

            piMoved = 0;

            if (iTable > 1)
            {
#if SQLITE_OMIT_AUTOVACUUM
freePage(pPage, ref rc);
releasePage(pPage);
#else
                if (pBt.autoVacuum)
                {
                    Pgno maxRootPgno = 0;
                    sqlite3BtreeGetMeta(p, BTREE_LARGEST_ROOT_PAGE, ref maxRootPgno);

                    if (iTable == maxRootPgno)
                    {
                        /* If the table being dropped is the table with the largest root-page
                        ** number in the database, put the root page on the free list.
                        */
                        freePage(pPage, ref rc);
                        releasePage(pPage);
                        if (rc != SQLITE_OK)
                        {
                            return rc;
                        }
                    }
                    else
                    {
                        /* The table being dropped does not have the largest root-page
                        ** number in the database. So move the page that does into the
                        ** gap left by the deleted root-page.
                        */
                        MemPage pMove = new MemPage();
                        releasePage(pPage);
                        rc = btreeGetPage(pBt, maxRootPgno, ref pMove, 0);
                        if (rc != SQLITE_OK)
                        {
                            return rc;
                        }
                        rc = relocatePage(pBt, pMove, PTRMAP_ROOTPAGE, 0, iTable, 0);
                        releasePage(pMove);
                        if (rc != SQLITE_OK)
                        {
                            return rc;
                        }
                        pMove = null;
                        rc = btreeGetPage(pBt, maxRootPgno, ref pMove, 0);
                        freePage(pMove, ref rc);
                        releasePage(pMove);
                        if (rc != SQLITE_OK)
                        {
                            return rc;
                        }
                        piMoved = (int)maxRootPgno;
                    }

                    /* Set the new 'max-root-page' value in the database header. This
                    ** is the old value less one, less one more if that happens to
                    ** be a root-page number, less one again if that is the
                    ** PENDING_BYTE_PAGE.
                    */
                    maxRootPgno--;
                    while (maxRootPgno == PENDING_BYTE_PAGE(pBt)
                    || PTRMAP_ISPAGE(pBt, maxRootPgno))
                    {
                        maxRootPgno--;
                    }
                    Debug.Assert(maxRootPgno != PENDING_BYTE_PAGE(pBt));

                    rc = sqlite3BtreeUpdateMeta(p, 4, maxRootPgno);
                }
                else
                {
                    freePage(pPage, ref rc);
                    releasePage(pPage);
                }
#endif
            }
            else
            {
                /* If sqlite3BtreeDropTable was called on page 1.
                ** This really never should happen except in a corrupt
                ** database.
                */
                zeroPage(pPage, PTF_INTKEY | PTF_LEAF);
                releasePage(pPage);
            }
            return rc;
        }
        static int sqlite3BtreeDropTable(Btree p, int iTable, ref int piMoved)
        {
            int rc;
            sqlite3BtreeEnter(p);
            rc = btreeDropTable(p, (u32)iTable, ref piMoved);
            sqlite3BtreeLeave(p);
            return rc;
        }


        /*
        ** This function may only be called if the b-tree connection already
        ** has a read or write transaction open on the database.
        **
        ** Read the meta-information out of a database file.  Meta[0]
        ** is the number of free pages currently in the database.  Meta[1]
        ** through meta[15] are available for use by higher layers.  Meta[0]
        ** is read-only, the others are read/write.
        **
        ** The schema layer numbers meta values differently.  At the schema
        ** layer (and the SetCookie and ReadCookie opcodes) the number of
        ** free pages is not visible.  So Cookie[0] is the same as Meta[1].
        */
        static void sqlite3BtreeGetMeta(Btree p, int idx, ref u32 pMeta)
        {
            BtShared pBt = p.pBt;

            sqlite3BtreeEnter(p);
            Debug.Assert(p.inTrans > TRANS_NONE);
            Debug.Assert(SQLITE_OK == querySharedCacheTableLock(p, MASTER_ROOT, READ_LOCK));
            Debug.Assert(pBt.pPage1 != null);
            Debug.Assert(idx >= 0 && idx <= 15);

            pMeta = sqlite3Get4byte(pBt.pPage1.aData, 36 + idx * 4);

            /* If auto-vacuum is disabled in this build and this is an auto-vacuum
            ** database, mark the database as read-only.  */
#if SQLITE_OMIT_AUTOVACUUM
if( idx==BTREE_LARGEST_ROOT_PAGE && pMeta>0 ) pBt.readOnly = 1;
#endif

            sqlite3BtreeLeave(p);
        }

        /*
        ** Write meta-information back into the database.  Meta[0] is
        ** read-only and may not be written.
        */
        static int sqlite3BtreeUpdateMeta(Btree p, int idx, u32 iMeta)
        {
            BtShared pBt = p.pBt;
            byte[] pP1;
            int rc;
            Debug.Assert(idx >= 1 && idx <= 15);
            sqlite3BtreeEnter(p);
            Debug.Assert(p.inTrans == TRANS_WRITE);
            Debug.Assert(pBt.pPage1 != null);
            pP1 = pBt.pPage1.aData;
            rc = sqlite3PagerWrite(pBt.pPage1.pDbPage);
            if (rc == SQLITE_OK)
            {
                sqlite3Put4byte(pP1, 36 + idx * 4, iMeta);
#if !SQLITE_OMIT_AUTOVACUUM
                if (idx == BTREE_INCR_VACUUM)
                {
                    Debug.Assert(pBt.autoVacuum || iMeta == 0);
                    Debug.Assert(iMeta == 0 || iMeta == 1);
                    pBt.incrVacuum = iMeta != 0;
                }
#endif
            }
            sqlite3BtreeLeave(p);
            return rc;
        }

#if !SQLITE_OMIT_BTREECOUNT
        /*
** The first argument, pCur, is a cursor opened on some b-tree. Count the
** number of entries in the b-tree and write the result to pnEntry.
**
** SQLITE_OK is returned if the operation is successfully executed.
** Otherwise, if an error is encountered (i.e. an IO error or database
** corruption) an SQLite error code is returned.
*/
        static int sqlite3BtreeCount(BtCursor pCur, ref i64 pnEntry)
        {
            i64 nEntry = 0;                      /* Value to return in pnEntry */
            int rc;                              /* Return code */
            rc = moveToRoot(pCur);

            /* Unless an error occurs, the following loop runs one iteration for each
            ** page in the B-Tree structure (not including overflow pages).
            */
            while (rc == SQLITE_OK)
            {
                int iIdx;                          /* Index of child node in parent */
                MemPage pPage;                    /* Current page of the b-tree */

                /* If this is a leaf page or the tree is not an int-key tree, then
                ** this page contains countable entries. Increment the entry counter
                ** accordingly.
                */
                pPage = pCur.apPage[pCur.iPage];
                if (pPage.leaf != 0 || 0 == pPage.intKey)
                {
                    nEntry += pPage.nCell;
                }

                /* pPage is a leaf node. This loop navigates the cursor so that it
                ** points to the first interior cell that it points to the parent of
                ** the next page in the tree that has not yet been visited. The
                ** pCur.aiIdx[pCur.iPage] value is set to the index of the parent cell
                ** of the page, or to the number of cells in the page if the next page
                ** to visit is the right-child of its parent.
                **
                ** If all pages in the tree have been visited, return SQLITE_OK to the
                ** caller.
                */
                if (pPage.leaf != 0)
                {
                    do
                    {
                        if (pCur.iPage == 0)
                        {
                            /* All pages of the b-tree have been visited. Return successfully. */
                            pnEntry = nEntry;
                            return SQLITE_OK;
                        }
                        moveToParent(pCur);
                    } while (pCur.aiIdx[pCur.iPage] >= pCur.apPage[pCur.iPage].nCell);

                    pCur.aiIdx[pCur.iPage]++;
                    pPage = pCur.apPage[pCur.iPage];
                }

                /* Descend to the child node of the cell that the cursor currently
                ** points at. This is the right-child if (iIdx==pPage.nCell).
                */
                iIdx = pCur.aiIdx[pCur.iPage];
                if (iIdx == pPage.nCell)
                {
                    rc = moveToChild(pCur, sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8));
                }
                else
                {
                    rc = moveToChild(pCur, sqlite3Get4byte(pPage.aData, findCell(pPage, iIdx)));
                }
            }

            /* An error has occurred. Return an error code. */
            return rc;
        }
#endif

        /*
** Return the pager associated with a BTree.  This routine is used for
** testing and debugging only.
*/
        static Pager sqlite3BtreePager(Btree p)
        {
            return p.pBt.pPager;
        }

#if !SQLITE_OMIT_INTEGRITY_CHECK
        /*
** Append a message to the error message string.
*/
        static void checkAppendMsg(
        IntegrityCk pCheck,
        string zMsg1,
        string zFormat,
        params object[] ap
        )
        {
            if (0 == pCheck.mxErr)
                return;
            //va_list ap;
            lock (lock_va_list)
            {
                pCheck.mxErr--;
                pCheck.nErr++;
                va_start(ap, zFormat);
                if (pCheck.errMsg.zText.Length != 0)
                {
                    sqlite3StrAccumAppend(pCheck.errMsg, "\n", 1);
                }
                if (zMsg1.Length > 0)
                {
                    sqlite3StrAccumAppend(pCheck.errMsg, zMsg1.ToString(), -1);
                }
                sqlite3VXPrintf(pCheck.errMsg, 1, zFormat, ap);
                va_end(ref ap);
            }
        }

        static void checkAppendMsg(
        IntegrityCk pCheck,
        StringBuilder zMsg1,
        string zFormat,
        params object[] ap
        )
        {
            if (0 == pCheck.mxErr)
                return;
            //va_list ap;
            lock (lock_va_list)
            {
                pCheck.mxErr--;
                pCheck.nErr++;
                va_start(ap, zFormat);
                if (pCheck.errMsg.zText.Length != 0)
                {
                    sqlite3StrAccumAppend(pCheck.errMsg, "\n", 1);
                }
                if (zMsg1.Length > 0)
                {
                    sqlite3StrAccumAppend(pCheck.errMsg, zMsg1.ToString(), -1);
                }
                sqlite3VXPrintf(pCheck.errMsg, 1, zFormat, ap);
                va_end(ref ap);
            }
            //if( pCheck.errMsg.mallocFailed ){
            //  pCheck.mallocFailed = 1;
            //}
        }
#endif //* SQLITE_OMIT_INTEGRITY_CHECK */

#if !SQLITE_OMIT_INTEGRITY_CHECK
        /*
** Add 1 to the reference count for page iPage.  If this is the second
** reference to the page, add an error message to pCheck.zErrMsg.
** Return 1 if there are 2 ore more references to the page and 0 if
** if this is the first reference to the page.
**
** Also check that the page number is in bounds.
*/
        static int checkRef(IntegrityCk pCheck, Pgno iPage, string zContext)
        {
            if (iPage == 0)
                return 1;
            if (iPage > pCheck.nPage)
            {
                checkAppendMsg(pCheck, zContext, "invalid page number %d", iPage);
                return 1;
            }
            if (pCheck.anRef[iPage] == 1)
            {
                checkAppendMsg(pCheck, zContext, "2nd reference to page %d", iPage);
                return 1;
            }
            return ((pCheck.anRef[iPage]++) > 1) ? 1 : 0;
        }

#if !SQLITE_OMIT_AUTOVACUUM
        /*
** Check that the entry in the pointer-map for page iChild maps to
** page iParent, pointer type ptrType. If not, append an error message
** to pCheck.
*/
        static void checkPtrmap(
        IntegrityCk pCheck,    /* Integrity check context */
        Pgno iChild,           /* Child page number */
        u8 eType,              /* Expected pointer map type */
        Pgno iParent,          /* Expected pointer map parent page number */
        string zContext /* Context description (used for error msg) */
        )
        {
            int rc;
            u8 ePtrmapType = 0;
            Pgno iPtrmapParent = 0;

            rc = ptrmapGet(pCheck.pBt, iChild, ref ePtrmapType, ref iPtrmapParent);
            if (rc != SQLITE_OK)
            {
                //if( rc==SQLITE_NOMEM || rc==SQLITE_IOERR_NOMEM ) pCheck.mallocFailed = 1;
                checkAppendMsg(pCheck, zContext, "Failed to read ptrmap key=%d", iChild);
                return;
            }

            if (ePtrmapType != eType || iPtrmapParent != iParent)
            {
                checkAppendMsg(pCheck, zContext,
                "Bad ptr map entry key=%d expected=(%d,%d) got=(%d,%d)",
                iChild, eType, iParent, ePtrmapType, iPtrmapParent);
            }
        }
#endif

        /*
** Check the integrity of the freelist or of an overflow page list.
** Verify that the number of pages on the list is N.
*/
        static void checkList(
        IntegrityCk pCheck,  /* Integrity checking context */
        int isFreeList,       /* True for a freelist.  False for overflow page list */
        int iPage,            /* Page number for first page in the list */
        int N,                /* Expected number of pages in the list */
        string zContext        /* Context for error messages */
        )
        {
            int i;
            int expected = N;
            int iFirst = iPage;
            while (N-- > 0 && pCheck.mxErr != 0)
            {
                PgHdr pOvflPage = new PgHdr();
                byte[] pOvflData;
                if (iPage < 1)
                {
                    checkAppendMsg(pCheck, zContext,
                    "%d of %d pages missing from overflow list starting at %d",
                    N + 1, expected, iFirst);
                    break;
                }
                if (checkRef(pCheck, (u32)iPage, zContext) != 0)
                    break;
                if (sqlite3PagerGet(pCheck.pPager, (Pgno)iPage, ref pOvflPage) != 0)
                {
                    checkAppendMsg(pCheck, zContext, "failed to get page %d", iPage);
                    break;
                }
                pOvflData = sqlite3PagerGetData(pOvflPage);
                if (isFreeList != 0)
                {
                    int n = (int)sqlite3Get4byte(pOvflData, 4);
#if !SQLITE_OMIT_AUTOVACUUM
                    if (pCheck.pBt.autoVacuum)
                    {
                        checkPtrmap(pCheck, (u32)iPage, PTRMAP_FREEPAGE, 0, zContext);
                    }
#endif
                    if (n > (int)pCheck.pBt.usableSize / 4 - 2)
                    {
                        checkAppendMsg(pCheck, zContext,
                        "freelist leaf count too big on page %d", iPage);
                        N--;
                    }
                    else
                    {
                        for (i = 0; i < n; i++)
                        {
                            Pgno iFreePage = sqlite3Get4byte(pOvflData, 8 + i * 4);
#if !SQLITE_OMIT_AUTOVACUUM
                            if (pCheck.pBt.autoVacuum)
                            {
                                checkPtrmap(pCheck, iFreePage, PTRMAP_FREEPAGE, 0, zContext);
                            }
#endif
                            checkRef(pCheck, iFreePage, zContext);
                        }
                        N -= n;
                    }
                }
#if !SQLITE_OMIT_AUTOVACUUM
                else
                {
                    /* If this database supports auto-vacuum and iPage is not the last
                    ** page in this overflow list, check that the pointer-map entry for
                    ** the following page matches iPage.
                    */
                    if (pCheck.pBt.autoVacuum && N > 0)
                    {
                        i = (int)sqlite3Get4byte(pOvflData);
                        checkPtrmap(pCheck, (u32)i, PTRMAP_OVERFLOW2, (u32)iPage, zContext);
                    }
                }
#endif
                iPage = (int)sqlite3Get4byte(pOvflData);
                sqlite3PagerUnref(pOvflPage);
            }
        }
#endif //* SQLITE_OMIT_INTEGRITY_CHECK */

#if !SQLITE_OMIT_INTEGRITY_CHECK
        /*
** Do various sanity checks on a single page of a tree.  Return
** the tree depth.  Root pages return 0.  Parents of root pages
** return 1, and so forth.
**
** These checks are done:
**
**      1.  Make sure that cells and freeblocks do not overlap
**          but combine to completely cover the page.
**  NO  2.  Make sure cell keys are in order.
**  NO  3.  Make sure no key is less than or equal to zLowerBound.
**  NO  4.  Make sure no key is greater than or equal to zUpperBound.
**      5.  Check the integrity of overflow pages.
**      6.  Recursively call checkTreePage on all children.
**      7.  Verify that the depth of all children is the same.
**      8.  Make sure this page is at least 33% full or else it is
**          the root of the tree.
*/

        static i64 refNULL = 0;   //Dummy for C# ref NULL

        static int checkTreePage(
        IntegrityCk pCheck,    /* Context for the sanity check */
        int iPage,             /* Page number of the page to check */
        string zParentContext, /* Parent context */
        ref i64 pnParentMinKey,
        ref i64 pnParentMaxKey,
        object _pnParentMinKey, /* C# Needed to determine if content passed*/
        object _pnParentMaxKey  /* C# Needed to determine if content passed*/
        )
        {
            MemPage pPage = new MemPage();
            int i, rc, depth, d2, pgno, cnt;
            int hdr, cellStart;
            int nCell;
            u8[] data;
            BtShared pBt;
            int usableSize;
            StringBuilder zContext = new StringBuilder(100);
            byte[] hit = null;
            i64 nMinKey = 0;
            i64 nMaxKey = 0;


            sqlite3_snprintf(200, zContext, "Page %d: ", iPage);

            /* Check that the page exists
            */
            pBt = pCheck.pBt;
            usableSize = (int)pBt.usableSize;
            if (iPage == 0)
                return 0;
            if (checkRef(pCheck, (u32)iPage, zParentContext) != 0)
                return 0;
            if ((rc = btreeGetPage(pBt, (Pgno)iPage, ref pPage, 0)) != 0)
            {
                checkAppendMsg(pCheck, zContext.ToString(),
                "unable to get the page. error code=%d", rc);
                return 0;
            }

            /* Clear MemPage.isInit to make sure the corruption detection code in
            ** btreeInitPage() is executed.  */
            pPage.isInit = 0;
            if ((rc = btreeInitPage(pPage)) != 0)
            {
                Debug.Assert(rc == SQLITE_CORRUPT);  /* The only possible error from InitPage */
                checkAppendMsg(pCheck, zContext.ToString(),
                "btreeInitPage() returns error code %d", rc);
                releasePage(pPage);
                return 0;
            }

            /* Check out all the cells.
            */
            depth = 0;
            for (i = 0; i < pPage.nCell && pCheck.mxErr != 0; i++)
            {
                u8[] pCell;
                u32 sz;
                CellInfo info = new CellInfo();

                /* Check payload overflow pages
                */
                sqlite3_snprintf(200, zContext,
                "On tree page %d cell %d: ", iPage, i);
                int iCell = findCell(pPage, i); //pCell = findCell( pPage, i );
                pCell = pPage.aData;
                btreeParseCellPtr(pPage, iCell, ref info); //btreeParseCellPtr( pPage, pCell, info );
                sz = info.nData;
                if (0 == pPage.intKey)
                    sz += (u32)info.nKey;
                /* For intKey pages, check that the keys are in order.
                */
                else if (i == 0)
                    nMinKey = nMaxKey = info.nKey;
                else
                {
                    if (info.nKey <= nMaxKey)
                    {
                        checkAppendMsg(pCheck, zContext.ToString(),
                        "Rowid %lld out of order (previous was %lld)", info.nKey, nMaxKey);
                    }
                    nMaxKey = info.nKey;
                }
                Debug.Assert(sz == info.nPayload);
                if ((sz > info.nLocal)
                    //&& (pCell[info.iOverflow]<=&pPage.aData[pBt.usableSize])
                )
                {
                    int nPage = (int)(sz - info.nLocal + usableSize - 5) / (usableSize - 4);
                    Pgno pgnoOvfl = sqlite3Get4byte(pCell, iCell, info.iOverflow);
#if !SQLITE_OMIT_AUTOVACUUM
                    if (pBt.autoVacuum)
                    {
                        checkPtrmap(pCheck, pgnoOvfl, PTRMAP_OVERFLOW1, (u32)iPage, zContext.ToString());
                    }
#endif
                    checkList(pCheck, 0, (int)pgnoOvfl, nPage, zContext.ToString());
                }

                /* Check sanity of left child page.
                */
                if (0 == pPage.leaf)
                {
                    pgno = (int)sqlite3Get4byte(pCell, iCell); //sqlite3Get4byte( pCell );
#if !SQLITE_OMIT_AUTOVACUUM
                    if (pBt.autoVacuum)
                    {
                        checkPtrmap(pCheck, (u32)pgno, PTRMAP_BTREE, (u32)iPage, zContext.ToString());
                    }
#endif
                    if (i == 0)
                        d2 = checkTreePage(pCheck, pgno, zContext.ToString(), ref nMinKey, ref refNULL, pCheck, null);
                    else
                        d2 = checkTreePage(pCheck, pgno, zContext.ToString(), ref nMinKey, ref nMaxKey, pCheck, pCheck);

                    if (i > 0 && d2 != depth)
                    {
                        checkAppendMsg(pCheck, zContext, "Child page depth differs");
                    }
                    depth = d2;
                }
            }
            if (0 == pPage.leaf)
            {
                pgno = (int)sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                sqlite3_snprintf(200, zContext,
                "On page %d at right child: ", iPage);
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum)
                {
                    checkPtrmap(pCheck, (u32)pgno, PTRMAP_BTREE, (u32)iPage, zContext.ToString());
                }
#endif
                //    checkTreePage(pCheck, pgno, zContext, NULL, !pPage->nCell ? NULL : &nMaxKey);
                if (0 == pPage.nCell)
                    checkTreePage(pCheck, pgno, zContext.ToString(), ref refNULL, ref refNULL, null, null);
                else
                    checkTreePage(pCheck, pgno, zContext.ToString(), ref refNULL, ref nMaxKey, null, pCheck);
            }

            /* For intKey leaf pages, check that the min/max keys are in order
            ** with any left/parent/right pages.
            */
            if (pPage.leaf != 0 && pPage.intKey != 0)
            {
                /* if we are a left child page */
                if (_pnParentMinKey != null)
                {
                    /* if we are the left most child page */
                    if (_pnParentMaxKey == null)
                    {
                        if (nMaxKey > pnParentMinKey)
                        {
                            checkAppendMsg(pCheck, zContext,
                            "Rowid %lld out of order (max larger than parent min of %lld)",
                            nMaxKey, pnParentMinKey);
                        }
                    }
                    else
                    {
                        if (nMinKey <= pnParentMinKey)
                        {
                            checkAppendMsg(pCheck, zContext,
                            "Rowid %lld out of order (min less than parent min of %lld)",
                            nMinKey, pnParentMinKey);
                        }
                        if (nMaxKey > pnParentMaxKey)
                        {
                            checkAppendMsg(pCheck, zContext,
                            "Rowid %lld out of order (max larger than parent max of %lld)",
                            nMaxKey, pnParentMaxKey);
                        }
                        pnParentMinKey = nMaxKey;
                    }
                    /* else if we're a right child page */
                }
                else if (_pnParentMaxKey != null)
                {
                    if (nMinKey <= pnParentMaxKey)
                    {
                        checkAppendMsg(pCheck, zContext,
                        "Rowid %lld out of order (min less than parent max of %lld)",
                        nMinKey, pnParentMaxKey);
                    }
                }
            }

            /* Check for complete coverage of the page
            */
            data = pPage.aData;
            hdr = pPage.hdrOffset;
            hit = sqlite3Malloc(pBt.pageSize);
            //if( hit==null ){
            //  pCheck.mallocFailed = 1;
            //}else
            {
                int contentOffset = get2byteNotZero(data, hdr + 5);
                Debug.Assert(contentOffset <= usableSize);  /* Enforced by btreeInitPage() */
                Array.Clear(hit, contentOffset, usableSize - contentOffset);//memset(hit+contentOffset, 0, usableSize-contentOffset);
                for (int iLoop = contentOffset - 1; iLoop >= 0; iLoop--)
                    hit[iLoop] = 1;//memset(hit, 1, contentOffset);
                nCell = get2byte(data, hdr + 3);
                cellStart = hdr + 12 - 4 * pPage.leaf;
                for (i = 0; i < nCell; i++)
                {
                    int pc = get2byte(data, cellStart + i * 2);
                    u32 size = 65536;
                    int j;
                    if (pc <= usableSize - 4)
                    {
                        size = cellSizePtr(pPage, data, pc);
                    }
                    if ((int)(pc + size - 1) >= usableSize)
                    {
                        checkAppendMsg(pCheck, "",
                        "Corruption detected in cell %d on page %d", i, iPage);
                    }
                    else
                    {
                        for (j = (int)(pc + size - 1); j >= pc; j--)
                            hit[j]++;
                    }
                }
                i = get2byte(data, hdr + 1);
                while (i > 0)
                {
                    int size, j;
                    Debug.Assert(i <= usableSize - 4);     /* Enforced by btreeInitPage() */
                    size = get2byte(data, i + 2);
                    Debug.Assert(i + size <= usableSize);  /* Enforced by btreeInitPage() */
                    for (j = i + size - 1; j >= i; j--)
                        hit[j]++;
                    j = get2byte(data, i);
                    Debug.Assert(j == 0 || j > i + size);  /* Enforced by btreeInitPage() */
                    Debug.Assert(j <= usableSize - 4);   /* Enforced by btreeInitPage() */
                    i = j;
                }
                for (i = cnt = 0; i < usableSize; i++)
                {
                    if (hit[i] == 0)
                    {
                        cnt++;
                    }
                    else if (hit[i] > 1)
                    {
                        checkAppendMsg(pCheck, "",
                        "Multiple uses for byte %d of page %d", i, iPage);
                        break;
                    }
                }
                if (cnt != data[hdr + 7])
                {
                    checkAppendMsg(pCheck, "",
                    "Fragmentation of %d bytes reported as %d on page %d",
                    cnt, data[hdr + 7], iPage);
                }
            }
            sqlite3PageFree(ref hit);
            releasePage(pPage);
            return depth + 1;
        }
#endif //* SQLITE_OMIT_INTEGRITY_CHECK */

#if !SQLITE_OMIT_INTEGRITY_CHECK
        /*
** This routine does a complete check of the given BTree file.  aRoot[] is
** an array of pages numbers were each page number is the root page of
** a table.  nRoot is the number of entries in aRoot.
**
** A read-only or read-write transaction must be opened before calling
** this function.
**
** Write the number of error seen in pnErr.  Except for some memory
** allocation errors,  an error message held in memory obtained from
** malloc is returned if pnErr is non-zero.  If pnErr==null then NULL is
** returned.  If a memory allocation error occurs, NULL is returned.
*/
        static string sqlite3BtreeIntegrityCheck(
        Btree p,       /* The btree to be checked */
        int[] aRoot,   /* An array of root pages numbers for individual trees */
        int nRoot,     /* Number of entries in aRoot[] */
        int mxErr,     /* Stop reporting errors after this many */
        ref int pnErr  /* Write number of errors seen to this variable */
        )
        {
            Pgno i;
            int nRef;
            IntegrityCk sCheck = new IntegrityCk();
            BtShared pBt = p.pBt;
            StringBuilder zErr = new StringBuilder(100);//char zErr[100];

            sqlite3BtreeEnter(p);
            Debug.Assert(p.inTrans > TRANS_NONE && pBt.inTransaction > TRANS_NONE);
            nRef = sqlite3PagerRefcount(pBt.pPager);
            sCheck.pBt = pBt;
            sCheck.pPager = pBt.pPager;
            sCheck.nPage = btreePagecount(sCheck.pBt);
            sCheck.mxErr = mxErr;
            sCheck.nErr = 0;
            //sCheck.mallocFailed = 0;
            pnErr = 0;
            if (sCheck.nPage == 0)
            {
                sqlite3BtreeLeave(p);
                return "";
            }
            sCheck.anRef = sqlite3Malloc(sCheck.anRef, (int)sCheck.nPage + 1);
            //if( !sCheck.anRef ){
            //  pnErr = 1;
            //  sqlite3BtreeLeave(p);
            //  return 0;
            //}
            // for (i = 0; i <= sCheck.nPage; i++) { sCheck.anRef[i] = 0; }
            i = PENDING_BYTE_PAGE(pBt);
            if (i <= sCheck.nPage)
            {
                sCheck.anRef[i] = 1;
            }
            sqlite3StrAccumInit(sCheck.errMsg, null, 1000, 20000);
            //sCheck.errMsg.useMalloc = 2;

            /* Check the integrity of the freelist
            */
            checkList(sCheck, 1, (int)sqlite3Get4byte(pBt.pPage1.aData, 32),
            (int)sqlite3Get4byte(pBt.pPage1.aData, 36), "Main freelist: ");

            /* Check all the tables.
            */
            for (i = 0; (int)i < nRoot && sCheck.mxErr != 0; i++)
            {
                if (aRoot[i] == 0)
                    continue;
#if !SQLITE_OMIT_AUTOVACUUM
                if (pBt.autoVacuum && aRoot[i] > 1)
                {
                    checkPtrmap(sCheck, (u32)aRoot[i], PTRMAP_ROOTPAGE, 0, "");
                }
#endif
                checkTreePage(sCheck, aRoot[i], "List of tree roots: ", ref refNULL, ref refNULL, null, null);
            }

            /* Make sure every page in the file is referenced
            */
            for (i = 1; i <= sCheck.nPage && sCheck.mxErr != 0; i++)
            {
#if SQLITE_OMIT_AUTOVACUUM
if( sCheck.anRef[i]==null ){
checkAppendMsg(sCheck, 0, "Page %d is never used", i);
}
#else
                /* If the database supports auto-vacuum, make sure no tables contain
** references to pointer-map pages.
*/
                if (sCheck.anRef[i] == 0 &&
                (PTRMAP_PAGENO(pBt, i) != i || !pBt.autoVacuum))
                {
                    checkAppendMsg(sCheck, "", "Page %d is never used", i);
                }
                if (sCheck.anRef[i] != 0 &&
                (PTRMAP_PAGENO(pBt, i) == i && pBt.autoVacuum))
                {
                    checkAppendMsg(sCheck, "", "Pointer map page %d is referenced", i);
                }
#endif
            }

            /* Make sure this analysis did not leave any unref() pages.
            ** This is an internal consistency check; an integrity check
            ** of the integrity check.
            */
            if (NEVER(nRef != sqlite3PagerRefcount(pBt.pPager)))
            {
                checkAppendMsg(sCheck, "",
                "Outstanding page count goes from %d to %d during this analysis",
                nRef, sqlite3PagerRefcount(pBt.pPager)
                );
            }

            /* Clean  up and report errors.
            */
            sqlite3BtreeLeave(p);
            sCheck.anRef = null;// sqlite3_free( ref sCheck.anRef );
            //if( sCheck.mallocFailed ){
            //  sqlite3StrAccumReset(sCheck.errMsg);
            //  pnErr = sCheck.nErr+1;
            //  return 0;
            //}
            pnErr = sCheck.nErr;
            if (sCheck.nErr == 0)
                sqlite3StrAccumReset(sCheck.errMsg);
            return sqlite3StrAccumFinish(sCheck.errMsg);
        }
#endif //* SQLITE_OMIT_INTEGRITY_CHECK */

        /*
** Return the full pathname of the underlying database file.
**
** The pager filename is invariant as long as the pager is
** open so it is safe to access without the BtShared mutex.
*/
        static string sqlite3BtreeGetFilename(Btree p)
        {
            Debug.Assert(p.pBt.pPager != null);
            return sqlite3PagerFilename(p.pBt.pPager);
        }

        /*
        ** Return the pathname of the journal file for this database. The return
        ** value of this routine is the same regardless of whether the journal file
        ** has been created or not.
        **
        ** The pager journal filename is invariant as long as the pager is
        ** open so it is safe to access without the BtShared mutex.
        */
        static string sqlite3BtreeGetJournalname(Btree p)
        {
            Debug.Assert(p.pBt.pPager != null);
            return sqlite3PagerJournalname(p.pBt.pPager);
        }

        /*
        ** Return non-zero if a transaction is active.
        */
        static bool sqlite3BtreeIsInTrans(Btree p)
        {
            Debug.Assert(p == null || sqlite3_mutex_held(p.db.mutex));
            return (p != null && (p.inTrans == TRANS_WRITE));
        }

#if !SQLITE_OMIT_WAL
/*
** Run a checkpoint on the Btree passed as the first argument.
**
** Return SQLITE_LOCKED if this or any other connection has an open 
** transaction on the shared-cache the argument Btree is connected to.
**
** Parameter eMode is one of SQLITE_CHECKPOINT_PASSIVE, FULL or RESTART.
*/
static int sqlite3BtreeCheckpointBtree *p, int eMode, int *pnLog, int *pnCkpt){
int rc = SQLITE_OK;
if( p != null){
BtShared pBt = p.pBt;
sqlite3BtreeEnter(p);
if( pBt.inTransaction!=TRANS_NONE ){
rc = SQLITE_LOCKED;
}else{
rc = sqlite3PagerCheckpoint(pBt.pPager, eMode, pnLog, pnCkpt);
}
sqlite3BtreeLeave(p);
}
return rc;
}
#endif

        /*
** Return non-zero if a read (or write) transaction is active.
*/
        static bool sqlite3BtreeIsInReadTrans(Btree p)
        {
            Debug.Assert(p != null);
            Debug.Assert(sqlite3_mutex_held(p.db.mutex));
            return p.inTrans != TRANS_NONE;
        }

        static bool sqlite3BtreeIsInBackup(Btree p)
        {
            Debug.Assert(p != null);
            Debug.Assert(sqlite3_mutex_held(p.db.mutex));
            return p.nBackup != 0;
        }

        /*
        ** This function returns a pointer to a blob of memory associated with
        ** a single shared-btree. The memory is used by client code for its own
        ** purposes (for example, to store a high-level schema associated with
        ** the shared-btree). The btree layer manages reference counting issues.
        **
        ** The first time this is called on a shared-btree, nBytes bytes of memory
        ** are allocated, zeroed, and returned to the caller. For each subsequent
        ** call the nBytes parameter is ignored and a pointer to the same blob
        ** of memory returned.
        **
        ** If the nBytes parameter is 0 and the blob of memory has not yet been
        ** allocated, a null pointer is returned. If the blob has already been
        ** allocated, it is returned as normal.
        **
        ** Just before the shared-btree is closed, the function passed as the 
        ** xFree argument when the memory allocation was made is invoked on the 
        ** blob of allocated memory. The xFree function should not call sqlite3_free()
        ** on the memory, the btree layer does that.
        */
        static Schema sqlite3BtreeSchema(Btree p, int nBytes, dxFreeSchema xFree)
        {
            BtShared pBt = p.pBt;
            sqlite3BtreeEnter(p);
            if (null == pBt.pSchema && nBytes != 0)
            {
                pBt.pSchema = new Schema();//sqlite3DbMallocZero(0, nBytes);
                pBt.xFreeSchema = xFree;
            }
            sqlite3BtreeLeave(p);
            return pBt.pSchema;
        }

        /*
        ** Return SQLITE_LOCKED_SHAREDCACHE if another user of the same shared
        ** btree as the argument handle holds an exclusive lock on the
        ** sqlite_master table. Otherwise SQLITE_OK.
        */
        static int sqlite3BtreeSchemaLocked(Btree p)
        {
            int rc;
            Debug.Assert(sqlite3_mutex_held(p.db.mutex));
            sqlite3BtreeEnter(p);
            rc = querySharedCacheTableLock(p, MASTER_ROOT, READ_LOCK);
            Debug.Assert(rc == SQLITE_OK || rc == SQLITE_LOCKED_SHAREDCACHE);
            sqlite3BtreeLeave(p);
            return rc;
        }


#if !SQLITE_OMIT_SHARED_CACHE
/*
** Obtain a lock on the table whose root page is iTab.  The
** lock is a write lock if isWritelock is true or a read lock
** if it is false.
*/
int sqlite3BtreeLockTable(Btree p, int iTab, u8 isWriteLock){
int rc = SQLITE_OK;
Debug.Assert( p.inTrans!=TRANS_NONE );
if( p.sharable ){
u8 lockType = READ_LOCK + isWriteLock;
Debug.Assert( READ_LOCK+1==WRITE_LOCK );
Debug.Assert( isWriteLock==null || isWriteLock==1 );

sqlite3BtreeEnter(p);
rc = querySharedCacheTableLock(p, iTab, lockType);
if( rc==SQLITE_OK ){
rc = setSharedCacheTableLock(p, iTab, lockType);
}
sqlite3BtreeLeave(p);
}
return rc;
}
#endif

#if !SQLITE_OMIT_INCRBLOB
/*
** Argument pCsr must be a cursor opened for writing on an
** INTKEY table currently pointing at a valid table entry.
** This function modifies the data stored as part of that entry.
**
** Only the data content may only be modified, it is not possible to
** change the length of the data stored. If this function is called with
** parameters that attempt to write past the end of the existing data,
** no modifications are made and SQLITE_CORRUPT is returned.
*/
int sqlite3BtreePutData(BtCursor pCsr, u32 offset, u32 amt, void *z){
int rc;
Debug.Assert( cursorHoldsMutex(pCsr) );
Debug.Assert( sqlite3_mutex_held(pCsr.pBtree.db.mutex) );
Debug.Assert( pCsr.isIncrblobHandle );

rc = restoreCursorPosition(pCsr);
if( rc!=SQLITE_OK ){
return rc;
}
Debug.Assert( pCsr.eState!=CURSOR_REQUIRESEEK );
if( pCsr.eState!=CURSOR_VALID ){
return SQLITE_ABORT;
}

/* Check some assumptions:
**   (a) the cursor is open for writing,
**   (b) there is a read/write transaction open,
**   (c) the connection holds a write-lock on the table (if required),
**   (d) there are no conflicting read-locks, and
**   (e) the cursor points at a valid row of an intKey table.
*/
if( !pCsr.wrFlag ){
return SQLITE_READONLY;
}
Debug.Assert( !pCsr.pBt.readOnly && pCsr.pBt.inTransaction==TRANS_WRITE );
Debug.Assert( hasSharedCacheTableLock(pCsr.pBtree, pCsr.pgnoRoot, 0, 2) );
Debug.Assert( !hasReadConflicts(pCsr.pBtree, pCsr.pgnoRoot) );
Debug.Assert( pCsr.apPage[pCsr.iPage].intKey );

return accessPayload(pCsr, offset, amt, (byte[] *)z, 1);
}

/*
** Set a flag on this cursor to cache the locations of pages from the
** overflow list for the current row. This is used by cursors opened
** for incremental blob IO only.
**
** This function sets a flag only. The actual page location cache
** (stored in BtCursor.aOverflow[]) is allocated and used by function
** accessPayload() (the worker function for sqlite3BtreeData() and
** sqlite3BtreePutData()).
*/
static void sqlite3BtreeCacheOverflow(BtCursor pCur){
Debug.Assert( cursorHoldsMutex(pCur) );
Debug.Assert( sqlite3_mutex_held(pCur.pBtree.db.mutex) );
invalidateOverflowCache(pCur)
pCur.isIncrblobHandle = 1;
}
#endif

        /*
** Set both the "read version" (single byte at byte offset 18) and 
** "write version" (single byte at byte offset 19) fields in the database
** header to iVersion.
*/
        static int sqlite3BtreeSetVersion(Btree pBtree, int iVersion)
        {
            BtShared pBt = pBtree.pBt;
            int rc;                         /* Return code */

            Debug.Assert(pBtree.inTrans == TRANS_NONE);
            Debug.Assert(iVersion == 1 || iVersion == 2);

            /* If setting the version fields to 1, do not automatically open the
            ** WAL connection, even if the version fields are currently set to 2.
            */
            pBt.doNotUseWAL = iVersion == 1;

            rc = sqlite3BtreeBeginTrans(pBtree, 0);
            if (rc == SQLITE_OK)
            {
                u8[] aData = pBt.pPage1.aData;
                if (aData[18] != (u8)iVersion || aData[19] != (u8)iVersion)
                {
                    rc = sqlite3BtreeBeginTrans(pBtree, 2);
                    if (rc == SQLITE_OK)
                    {
                        rc = sqlite3PagerWrite(pBt.pPage1.pDbPage);
                        if (rc == SQLITE_OK)
                        {
                            aData[18] = (u8)iVersion;
                            aData[19] = (u8)iVersion;
                        }
                    }
                }
            }

            pBt.doNotUseWAL = false;
            return rc;
        }






















    }
}
