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
        internal bool cursorHoldsMutex() { return MutexEx.sqlite3_mutex_held(this.pBt.mutex); }
#else
        internal bool cursorHoldsMutex() { return true; }
#endif

        #region Properties

        internal int sqlite3BtreeCursorSize() { return -1; }
        internal void sqlite3BtreeCursorZero() { this.Clear(); }

        internal void sqlite3BtreeSetCachedRowid(long iRowid)
        {
            for (var p = this.pBt.pCursor; p != null; p = p.pNext)
                if (p.pgnoRoot == this.pgnoRoot)
                    p.cachedRowid = iRowid;
            Debug.Assert(this.cachedRowid == iRowid);
        }

        internal long sqlite3BtreeGetCachedRowid() { return this.cachedRowid; }

        #endregion

        internal SQLITE saveCursorPosition()
        {
            Debug.Assert(this.eState == CURSOR.VALID);
            Debug.Assert(this.pKey == null);
            Debug.Assert(cursorHoldsMutex());
            var rc = sqlite3BtreeKeySize(ref this.nKey);
            Debug.Assert(rc == SQLITE.OK);  // KeySize() cannot fail
            // If this is an intKey table, then the above call to BtreeKeySize() stores the integer key in pCur.nKey. In this case this value is
            // all that is required. Otherwise, if pCur is not open on an intKey table, then malloc space for and store the pCur.nKey bytes of key data.
            if (this.apPage[0].intKey == 0)
            {
                var pKey = MallocEx.sqlite3Malloc((int)this.nKey);
                rc = sqlite3BtreeKey(0, (uint)this.nKey, pKey);
                if (rc == SQLITE.OK)
                    this.pKey = pKey;
            }
            Debug.Assert(this.apPage[0].intKey == 0 || this.pKey == null);
            if (rc == SQLITE.OK)
            {
                for (var i = 0; i <= this.iPage; i++)
                {
                    this.apPage[i].releasePage();
                    this.apPage[i] = null;
                }
                this.iPage = -1;
                this.eState = CURSOR.REQUIRESEEK;
            }
            invalidateOverflowCache();
            return rc;
        }

        internal void sqlite3BtreeClearCursor()
        {
            Debug.Assert(cursorHoldsMutex());
            MallocEx.sqlite3_free(ref this.pKey);
            this.eState = CURSOR.INVALID;
        }

        internal SQLITE btreeMoveto(byte[] pKey, long nKey, int bias, ref int pRes)
        {
            Btree.UnpackedRecord pIdxKey; // Unpacked index key
            var aSpace = new Btree.UnpackedRecord();
            if (pKey != null)
            {
                Debug.Assert(nKey == (long)(int)nKey);
                pIdxKey = sqlite3VdbeRecordUnpack(this.pKeyInfo, (int)nKey, pKey, aSpace, 16);
            }
            else
                pIdxKey = null;
            var rc = sqlite3BtreeMovetoUnpacked(pIdxKey, nKey, bias != 0 ? 1 : 0, ref pRes);
            if (pKey != null)
                sqlite3VdbeDeleteUnpackedRecord(pIdxKey);
            return rc;
        }

        internal SQLITE btreeRestoreCursorPosition()
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState >= CURSOR.REQUIRESEEK);
            if (this.eState == CURSOR.FAULT)
                return (SQLITE)this.skipNext;
            this.eState = CURSOR.INVALID;
            var rc = btreeMoveto(this.pKey, this.nKey, 0, ref this.skipNext);
            if (rc == SQLITE.OK)
            {
                this.pKey = null;
                Debug.Assert(this.eState == CURSOR.VALID || this.eState == CURSOR.INVALID);
            }
            return rc;
        }

        internal SQLITE restoreCursorPosition() { return (this.eState >= CURSOR.REQUIRESEEK ? btreeRestoreCursorPosition() : SQLITE.OK); }

        internal SQLITE sqlite3BtreeCursorHasMoved(ref int pHasMoved)
        {
            var rc = restoreCursorPosition();
            if (rc != SQLITE.OK)
            {
                pHasMoved = 1;
                return rc;
            }
            pHasMoved = (this.eState != CURSOR.VALID || this.skipNext != 0 ? 1 : 0);
            return SQLITE.OK;
        }

        internal SQLITE sqlite3BtreeCloseCursor()
        {
            var pBtree = this.pBtree;
            if (pBtree != null)
            {
                var pBt = this.pBt;
                pBtree.sqlite3BtreeEnter();
                sqlite3BtreeClearCursor();
                if (this.pPrev != null)
                    this.pPrev.pNext = this.pNext;
                else
                    pBt.pCursor = this.pNext;
                if (this.pNext != null)
                    this.pNext.pPrev = this.pPrev;
                for (var i = 0; i <= this.iPage; i++)
                    this.apPage[i].releasePage();
                pBt.unlockBtreeIfUnused();
                Btree.invalidateOverflowCache(this);
                pBtree.sqlite3BtreeLeave();
            }
            return SQLITE.OK;
        }

#if !NDEBUG
        internal void assertCellInfo()
        {
            var iPage = this.iPage;
            var info = new CellInfo();
            this.apPage[iPage].btreeParseCell(this.aiIdx[iPage], ref info);
            Debug.Assert(info.GetHashCode() == this.info.GetHashCode() || info.Equals(this.info));
        }
#else
        internal void assertCellInfo() { }
#endif

#if true //!_MSC_VER
        internal void getCellInfo()
        {
            if (this.info.nSize == 0)
            {
                var iPage = this.iPage;
                this.apPage[iPage].btreeParseCell(this.aiIdx[iPage], ref this.info);
                this.validNKey = true;
            }
            else
                assertCellInfo();
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
        internal bool sqlite3BtreeCursorIsValid() { return this != null && this.eState == CURSOR.VALID; }
#else
        internal bool sqlite3BtreeCursorIsValid() { return true; }
#endif

        internal SQLITE sqlite3BtreeKeySize(ref long pSize)
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.INVALID || this.eState == CURSOR.VALID);
            if (this.eState != CURSOR.VALID)
                pSize = 0;
            else
            {
                getCellInfo();
                pSize = this.info.nKey;
            }
            return SQLITE.OK;
        }

        internal SQLITE sqlite3BtreeDataSize(ref uint pSize)
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.VALID);
            getCellInfo();
            pSize = this.info.nData;
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

        internal SQLITE accessPayload(uint offset, uint amt, byte[] pBuf, int eOp)
        {
            uint pBufOffset = 0;
            byte[] aPayload;
            var rc = SQLITE.OK;
            int iIdx = 0;
            var pPage = this.apPage[this.iPage]; // Btree page of current entry
            var pBt = this.pBt;                  // Btree this cursor belongs to
            Debug.Assert(pPage != null);
            Debug.Assert(this.eState == CURSOR.VALID);
            Debug.Assert(this.aiIdx[this.iPage] < pPage.nCell);
            Debug.Assert(cursorHoldsMutex());
            getCellInfo();
            aPayload = this.info.pCell; //pCur.info.pCell + pCur.info.nHeader;
            var nKey = (uint)(pPage.intKey != 0 ? 0 : (int)this.info.nKey);
            if (Check.NEVER(offset + amt > nKey + this.info.nData) || this.info.nLocal > pBt.usableSize)
                // Trying to read or write past the end of the data is an error
                return SysEx.SQLITE_CORRUPT_BKPT();
            // Check if data must be read/written to/from the btree page itself.
            if (offset < this.info.nLocal)
            {
                var a = (int)amt;
                if (a + offset > this.info.nLocal)
                    a = (int)(this.info.nLocal - offset);
                rc = copyPayload(aPayload, (uint)(offset + this.info.iCell + this.info.nHeader), pBuf, pBufOffset, (uint)a, eOp, pPage.pDbPage);
                offset = 0;
                pBufOffset += (uint)a;
                amt -= (uint)a;
            }
            else
                offset -= this.info.nLocal;
            if (rc == SQLITE.OK && amt > 0)
            {
                var ovflSize = (uint)(pBt.usableSize - 4);  // Bytes content per ovfl page
                Pgno nextPage = ConvertEx.sqlite3Get4byte(aPayload, this.info.nLocal + this.info.iCell + this.info.nHeader);
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
                        rc = pBt.getOverflowPage(nextPage, out MemPageDummy, out nextPage);
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

        internal SQLITE sqlite3BtreeKey(uint offset, uint amt, byte[] pBuf)
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.VALID);
            Debug.Assert(this.iPage >= 0 && this.apPage[this.iPage] != null);
            Debug.Assert(this.aiIdx[this.iPage] < this.apPage[this.iPage].nCell);
            return accessPayload(offset, amt, pBuf, 0);
        }

        internal SQLITE sqlite3BtreeData(uint offset, uint amt, byte[] pBuf)
        {
#if !SQLITE_OMIT_INCRBLOB
            if (pCur.eState == CURSOR.INVALID)
                return SQLITE.ABORT;
#endif
            Debug.Assert(cursorHoldsMutex());
            var rc = restoreCursorPosition();
            if (rc == SQLITE.OK)
            {
                Debug.Assert(this.eState == CURSOR.VALID);
                Debug.Assert(this.iPage >= 0 && this.apPage[this.iPage] != null);
                Debug.Assert(this.aiIdx[this.iPage] < this.apPage[this.iPage].nCell);
                rc = accessPayload(offset, amt, pBuf, 0);
            }
            return rc;
        }

        internal byte[] fetchPayload(ref int pAmt, ref int outOffset, bool skipKey)
        {
            Debug.Assert(this != null && this.iPage >= 0 && this.apPage[this.iPage] != null);
            Debug.Assert(this.eState == CURSOR.VALID);
            Debug.Assert(cursorHoldsMutex());
            outOffset = -1;
            var pPage = this.apPage[this.iPage];
            Debug.Assert(this.aiIdx[this.iPage] < pPage.nCell);
            if (Check.NEVER(this.info.nSize == 0))
                this.apPage[this.iPage].btreeParseCell(this.aiIdx[this.iPage], ref this.info);
            var aPayload = MallocEx.sqlite3Malloc(this.info.nSize - this.info.nHeader);
            var nKey = (pPage.intKey != 0 ? 0 : (uint)this.info.nKey);
            uint nLocal;
            if (skipKey)
            {
                outOffset = (int)(this.info.iCell + this.info.nHeader + nKey);
                Buffer.BlockCopy(this.info.pCell, outOffset, aPayload, 0, (int)(this.info.nSize - this.info.nHeader - nKey));
                nLocal = this.info.nLocal - nKey;
            }
            else
            {
                outOffset = (int)(this.info.iCell + this.info.nHeader);
                Buffer.BlockCopy(this.info.pCell, outOffset, aPayload, 0, this.info.nSize - this.info.nHeader);
                nLocal = this.info.nLocal;
                Debug.Assert(nLocal <= nKey);
            }
            pAmt = (int)nLocal;
            return aPayload;
        }

        internal byte[] sqlite3BtreeKeyFetch(ref int pAmt, ref int outOffset)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBtree.db.mutex));
            Debug.Assert(cursorHoldsMutex());
            byte[] p = null;
            if (Check.ALWAYS(this.eState == CURSOR.VALID))
                p = fetchPayload(ref pAmt, ref outOffset, false);
            return p;
        }

        //TODO remove sqlite3BtreeKeyFetch or sqlite3BtreeDataFetch
        internal byte[] sqlite3BtreeDataFetch(ref int pAmt, ref int outOffset)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBtree.db.mutex));
            Debug.Assert(cursorHoldsMutex());
            byte[] p = null;
            if (Check.ALWAYS(this.eState == CURSOR.VALID))
                p = fetchPayload(ref pAmt, ref outOffset, true);
            return p;
        }

        internal SQLITE moveToChild(Pgno newPgno)
        {
            var i = this.iPage;
            var pNewPage = new MemPage();
            var pBt = this.pBt;
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.VALID);
            Debug.Assert(this.iPage < BTCURSOR_MAX_DEPTH);
            if (this.iPage >= (BTCURSOR_MAX_DEPTH - 1))
                return SysEx.SQLITE_CORRUPT_BKPT();
            var rc = pBt.getAndInitPage(newPgno, ref pNewPage);
            if (rc != SQLITE.OK)
                return rc;
            this.apPage[i + 1] = pNewPage;
            this.aiIdx[i + 1] = 0;
            this.iPage++;
            this.info.nSize = 0;
            this.validNKey = false;
            if (pNewPage.nCell < 1 || pNewPage.intKey != this.apPage[i].intKey)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return SQLITE.OK;
        }

        internal void moveToParent()
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.VALID);
            Debug.Assert(this.iPage > 0);
            Debug.Assert(this.apPage[this.iPage] != null);
            MemPage.assertParentIndex(this.apPage[this.iPage - 1], this.aiIdx[this.iPage - 1], this.apPage[this.iPage].pgno);
            this.apPage[this.iPage].releasePage();
            this.iPage--;
            this.info.nSize = 0;
            this.validNKey = false;
        }

        internal SQLITE moveToRoot()
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(CURSOR.INVALID < CURSOR.REQUIRESEEK);
            Debug.Assert(CURSOR.VALID < CURSOR.REQUIRESEEK);
            Debug.Assert(CURSOR.FAULT > CURSOR.REQUIRESEEK);
            if (this.eState >= CURSOR.REQUIRESEEK)
            {
                if (this.eState == CURSOR.FAULT)
                {
                    Debug.Assert(this.skipNext != 0);
                    return (SQLITE)this.skipNext;
                }
                sqlite3BtreeClearCursor();
            }
            var rc = SQLITE.OK;
            var p = this.pBtree;
            var pBt = p.pBt;
            if (this.iPage >= 0)
            {
                for (var i = 1; i <= this.iPage; i++)
                    this.apPage[i].releasePage();
                this.iPage = 0;
            }
            else
            {
                rc = pBt.getAndInitPage(this.pgnoRoot, ref this.apPage[0]);
                if (rc != SQLITE.OK)
                {
                    this.eState = CURSOR.INVALID;
                    return rc;
                }
                this.iPage = 0;
                // If pCur.pKeyInfo is not NULL, then the caller that opened this cursor expected to open it on an index b-tree. Otherwise, if pKeyInfo is
                // NULL, the caller expects a table b-tree. If this is not the case, return an SQLITE_CORRUPT error.
                Debug.Assert(this.apPage[0].intKey == 1 || this.apPage[0].intKey == 0);
                if ((this.pKeyInfo == null) != (this.apPage[0].intKey != 0))
                    return SysEx.SQLITE_CORRUPT_BKPT();
            }
            // Assert that the root page is of the correct type. This must be the case as the call to this function that loaded the root-page (either
            // this call or a previous invocation) would have detected corruption if the assumption were not true, and it is not possible for the flags
            // byte to have been modified while this cursor is holding a reference to the page.
            var pRoot = this.apPage[0];
            Debug.Assert(pRoot.pgno == this.pgnoRoot);
            Debug.Assert(pRoot.isInit != 0 && (this.pKeyInfo == null) == (pRoot.intKey != 0));
            this.aiIdx[0] = 0;
            this.info.nSize = 0;
            this.atLast = 0;
            this.validNKey = false;
            if (pRoot.nCell == 0 && 0 == pRoot.leaf)
            {
                if (pRoot.pgno != 1)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                Pgno subpage = ConvertEx.sqlite3Get4byte(pRoot.aData, pRoot.hdrOffset + 8);
                this.eState = CURSOR.VALID;
                rc = moveToChild(subpage);
            }
            else
                this.eState = (pRoot.nCell > 0 ? CURSOR.VALID : CURSOR.INVALID);
            return rc;
        }

        internal SQLITE moveToLeftmost()
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.VALID);
            MemPage pPage;
            var rc = SQLITE.OK;
            while (rc == SQLITE.OK && 0 == (pPage = this.apPage[this.iPage]).leaf)
            {
                Debug.Assert(this.aiIdx[this.iPage] < pPage.nCell);
                Pgno pgno = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.findCell(this.aiIdx[this.iPage]));
                rc = moveToChild(pgno);
            }
            return rc;
        }

        internal SQLITE moveToRightmost()
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(this.eState == CURSOR.VALID);
            var rc = SQLITE.OK;
            MemPage pPage = null;
            while (rc == SQLITE.OK && 0 == (pPage = this.apPage[this.iPage]).leaf)
            {
                Pgno pgno = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                this.aiIdx[this.iPage] = pPage.nCell;
                rc = moveToChild(pgno);
            }
            if (rc == SQLITE.OK)
            {
                this.aiIdx[this.iPage] = (ushort)(pPage.nCell - 1);
                this.info.nSize = 0;
                this.validNKey = false;
            }
            return rc;
        }

        internal SQLITE sqlite3BtreeFirst(ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBtree.db.mutex));
            var rc = moveToRoot();
            if (rc == SQLITE.OK)
                if (this.eState == CURSOR.INVALID)
                {
                    Debug.Assert(this.apPage[this.iPage].nCell == 0);
                    pRes = 1;
                }
                else
                {
                    Debug.Assert(this.apPage[this.iPage].nCell > 0);
                    pRes = 0;
                    rc = moveToLeftmost();
                }
            return rc;
        }

        internal SQLITE sqlite3BtreeLast(ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBtree.db.mutex));
            // If the cursor already points to the last entry, this is a no-op.
            if (this.eState == CURSOR.VALID && this.atLast != 0)
            {
#if DEBUG
                // This block serves to Debug.Assert() that the cursor really does point to the last entry in the b-tree.
                for (var ii = 0; ii < this.iPage; ii++)
                    Debug.Assert(this.aiIdx[ii] == this.apPage[ii].nCell);
                Debug.Assert(this.aiIdx[this.iPage] == this.apPage[this.iPage].nCell - 1);
                Debug.Assert(this.apPage[this.iPage].leaf != 0);
#endif
                return SQLITE.OK;
            }
            var rc = moveToRoot();
            if (rc == SQLITE.OK)
                if (this.eState == CURSOR.INVALID)
                {
                    Debug.Assert(this.apPage[this.iPage].nCell == 0);
                    pRes = 1;
                }
                else
                {
                    Debug.Assert(this.eState == CURSOR.VALID);
                    pRes = 0;
                    rc = moveToRightmost();
                    this.atLast = (byte)(rc == SQLITE.OK ? 1 : 0);
                }
            return rc;
        }

        internal SQLITE sqlite3BtreeMovetoUnpacked(Btree.UnpackedRecord pIdxKey, long intKey, int biasRight, ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex());
            Debug.Assert(MutexEx.sqlite3_mutex_held(this.pBtree.db.mutex));
            Debug.Assert((pIdxKey == null) == (this.pKeyInfo == null));
            // If the cursor is already positioned at the point we are trying to move to, then just return without doing any work
            if (this.eState == CURSOR.VALID && this.validNKey && this.apPage[0].intKey != 0)
            {
                if (this.info.nKey == intKey)
                {
                    pRes = 0;
                    return SQLITE.OK;
                }
                if (this.atLast != 0 && this.info.nKey < intKey)
                {
                    pRes = -1;
                    return SQLITE.OK;
                }
            }
            var rc = moveToRoot();
            if (rc != SQLITE.OK)
                return rc;
            Debug.Assert(this.apPage[this.iPage] != null);
            Debug.Assert(this.apPage[this.iPage].isInit != 0);
            Debug.Assert(this.apPage[this.iPage].nCell > 0 || this.eState == CURSOR.INVALID);
            if (this.eState == CURSOR.INVALID)
            {
                pRes = -1;
                Debug.Assert(this.apPage[this.iPage].nCell == 0);
                return SQLITE.OK;
            }
            Debug.Assert(this.apPage[0].intKey != 0 || pIdxKey != null);
            for (; ; )
            {
                Pgno chldPg;
                var pPage = this.apPage[this.iPage];
                // pPage.nCell must be greater than zero. If this is the root-page the cursor would have been INVALID above and this for(;;) loop
                // not run. If this is not the root-page, then the moveToChild() routine would have already detected db corruption. Similarly, pPage must
                // be the right kind (index or table) of b-tree page. Otherwise a moveToChild() or moveToRoot() call would have detected corruption.
                Debug.Assert(pPage.nCell > 0);
                Debug.Assert(pPage.intKey == ((pIdxKey == null) ? 1 : 0));
                var lwr = 0;
                var upr = pPage.nCell - 1;
                int idx;
                this.aiIdx[this.iPage] = (ushort)(biasRight != 0 ? (idx = upr) : (idx = (upr + lwr) / 2));
                int c;
                for (; ; )
                {
                    Debug.Assert(idx == this.aiIdx[this.iPage]);
                    this.info.nSize = 0;
                    int pCell = pPage.findCell(idx) + pPage.childPtrSize; // Pointer to current cell in pPage
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
                        this.validNKey = true;
                        this.info.nKey = nCellKey;
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
                            pPage.btreeParseCellPtr(pCellBody, ref this.info);
                            nCell = (int)this.info.nKey;
                            var pCellKey = MallocEx.sqlite3Malloc(nCell);
                            rc = accessPayload(0, (uint)nCell, pCellKey, 0);
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
                    this.aiIdx[this.iPage] = (ushort)(idx = (lwr + upr) / 2);
                }
                Debug.Assert(lwr == upr + 1);
                Debug.Assert(pPage.isInit != 0);
                if (pPage.leaf != 0)
                    chldPg = 0;
                else if (lwr >= pPage.nCell)
                    chldPg = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                else
                    chldPg = ConvertEx.sqlite3Get4byte(pPage.aData, pPage.findCell(lwr));
                if (chldPg == 0)
                {
                    Debug.Assert(this.aiIdx[this.iPage] < this.apPage[this.iPage].nCell);
                    pRes = c;
                    rc = SQLITE.OK;
                    goto moveto_finish;
                }
                this.aiIdx[this.iPage] = (ushort)lwr;
                this.info.nSize = 0;
                this.validNKey = false;
                rc = moveToChild(chldPg);
                if (rc != SQLITE.OK)
                    goto moveto_finish;
            }
        moveto_finish:
            return rc;
        }

        internal bool sqlite3BtreeEof()
        {
            // TODO: What if the cursor is in CURSOR_REQUIRESEEK but all table entries have been deleted? This API will need to change to return an error code
            // as well as the boolean result value.
            return (this.eState != CURSOR.VALID);
        }

        internal SQLITE sqlite3BtreeNext(ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex());
            var rc = restoreCursorPosition();
            if (rc != SQLITE.OK)
                return rc;
            if (this.eState == CURSOR.INVALID)
            {
                pRes = 1;
                return SQLITE.OK;
            }
            if (this.skipNext > 0)
            {
                this.skipNext = 0;
                pRes = 0;
                return SQLITE.OK;
            }
            this.skipNext = 0;
            //
            var pPage = this.apPage[this.iPage];
            var idx = ++this.aiIdx[this.iPage];
            Debug.Assert(pPage.isInit != 0);
            Debug.Assert(idx <= pPage.nCell);
            this.info.nSize = 0;
            this.validNKey = false;
            if (idx >= pPage.nCell)
            {
                if (pPage.leaf == 0)
                {
                    rc = moveToChild(ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8));
                    if (rc != SQLITE.OK)
                        return rc;
                    rc = moveToLeftmost();
                    pRes = 0;
                    return rc;
                }
                do
                {
                    if (this.iPage == 0)
                    {
                        pRes = 1;
                        this.eState = CURSOR.INVALID;
                        return SQLITE.OK;
                    }
                    moveToParent();
                    pPage = this.apPage[this.iPage];
                } while (this.aiIdx[this.iPage] >= pPage.nCell);
                pRes = 0;
                rc = (pPage.intKey != 0 ? sqlite3BtreeNext(ref pRes) : SQLITE.OK);
                return rc;
            }
            pRes = 0;
            if (pPage.leaf != 0)
                return SQLITE.OK;
            rc = moveToLeftmost();
            return rc;
        }

        internal SQLITE sqlite3BtreePrevious(ref int pRes)
        {
            Debug.Assert(cursorHoldsMutex());
            var rc = restoreCursorPosition();
            if (rc != SQLITE.OK)
                return rc;
            this.atLast = 0;
            if (this.eState == CURSOR.INVALID)
            {
                pRes = 1;
                return SQLITE.OK;
            }
            if (this.skipNext < 0)
            {
                this.skipNext = 0;
                pRes = 0;
                return SQLITE.OK;
            }
            this.skipNext = 0;
            //
            var pPage = this.apPage[this.iPage];
            Debug.Assert(pPage.isInit != 0);
            if (pPage.leaf == 0)
            {
                var idx = this.aiIdx[this.iPage];
                rc = moveToChild(ConvertEx.sqlite3Get4byte(pPage.aData, pPage.findCell(idx)));
                if (rc != SQLITE.OK)
                    return rc;
                rc = moveToRightmost();
            }
            else
            {
                while (this.aiIdx[this.iPage] == 0)
                {
                    if (this.iPage == 0)
                    {
                        this.eState = CURSOR.INVALID;
                        pRes = 1;
                        return SQLITE.OK;
                    }
                    moveToParent();
                }
                this.info.nSize = 0;
                this.validNKey = false;
                this.aiIdx[this.iPage]--;
                pPage = this.apPage[this.iPage];
                rc = (pPage.intKey != 0 && 0 == pPage.leaf ? sqlite3BtreePrevious(ref pRes) : SQLITE.OK);
            }
            pRes = 0;
            return rc;
        }

#if !SQLITE_OMIT_BTREECOUNT
        internal SQLITE sqlite3BtreeCount(ref long pnEntry)
        {
            long nEntry = 0; // Value to return in pnEntry
            var rc = moveToRoot();
            // Unless an error occurs, the following loop runs one iteration for each page in the B-Tree structure (not including overflow pages).
            while (rc == SQLITE.OK)
            {
                // If this is a leaf page or the tree is not an int-key tree, then this page contains countable entries. Increment the entry counter accordingly.
                var pPage = this.apPage[this.iPage]; // Current page of the b-tree 
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
                        if (this.iPage == 0)
                        {
                            // All pages of the b-tree have been visited. Return successfully.
                            pnEntry = nEntry;
                            return SQLITE.OK;
                        }
                        moveToParent();
                    } while (this.aiIdx[this.iPage] >= this.apPage[this.iPage].nCell);
                    this.aiIdx[this.iPage]++;
                    pPage = this.apPage[this.iPage];
                }
                // Descend to the child node of the cell that the cursor currently points at. This is the right-child if (iIdx==pPage.nCell).
                var iIdx = this.aiIdx[this.iPage]; // Index of child node in parent
                rc = (iIdx == pPage.nCell ? moveToChild(ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8)) : moveToChild(ConvertEx.sqlite3Get4byte(pPage.aData, pPage.findCell(iIdx))));
            }
            // An error has occurred. Return an error code.
            return rc;
        }
#endif
    }
}
