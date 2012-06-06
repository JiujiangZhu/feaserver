using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using SAVEPOINT = Contoso.Core.Pager.SAVEPOINT;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;

namespace Contoso.Core
{
    public partial class Btree
    {
#if false && !SQLITE_OMIT_INTEGRITY_CHECK

        internal static void checkAppendMsg(IntegrityCk pCheck, string zMsg1, string zFormat, params object[] ap)
        {
            if (pCheck.mxErr == 0)
                return;
            lock (lock_va_list)
            {
                pCheck.mxErr--;
                pCheck.nErr++;
                va_start(ap, zFormat);
                if (pCheck.errMsg.zText.Length != 0)
                    sqlite3StrAccumAppend(pCheck.errMsg, "\n", 1);
                if (zMsg1.Length > 0)
                    sqlite3StrAccumAppend(pCheck.errMsg, zMsg1.ToString(), -1);
                sqlite3VXPrintf(pCheck.errMsg, 1, zFormat, ap);
                va_end(ref ap);
            }
        }

        internal static void checkAppendMsg(IntegrityCk pCheck, StringBuilder zMsg1, string zFormat, params object[] ap)
        {
            if (pCheck.mxErr == 0)
                return;
            lock (lock_va_list)
            {
                pCheck.mxErr--;
                pCheck.nErr++;
                va_start(ap, zFormat);
                if (pCheck.errMsg.zText.Length != 0)
                    sqlite3StrAccumAppend(pCheck.errMsg, "\n", 1);
                if (zMsg1.Length > 0)
                    sqlite3StrAccumAppend(pCheck.errMsg, zMsg1.ToString(), -1);
                sqlite3VXPrintf(pCheck.errMsg, 1, zFormat, ap);
                va_end(ref ap);
            }
        }

        internal static int checkRef(IntegrityCk pCheck, Pgno iPage, string zContext)
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
        internal static void checkPtrmap(IntegrityCk pCheck, Pgno iChild, byte eType, Pgno iParent, string zContext)
        {
            byte ePtrmapType = 0;
            Pgno iPtrmapParent = 0;
            var rc = ptrmapGet(pCheck.pBt, iChild, ref ePtrmapType, ref iPtrmapParent);
            if (rc != SQLITE.OK)
            {
                checkAppendMsg(pCheck, zContext, "Failed to read ptrmap key=%d", iChild);
                return;
            }
            if (ePtrmapType != eType || iPtrmapParent != iParent)
                checkAppendMsg(pCheck, zContext, "Bad ptr map entry key=%d expected=(%d,%d) got=(%d,%d)", iChild, eType, iParent, ePtrmapType, iPtrmapParent);
        }
#endif

        internal static void checkList(IntegrityCk pCheck, int isFreeList, int iPage, int N, string zContext)
        {
            int i;
            int expected = N;
            int iFirst = iPage;
            while (N-- > 0 && pCheck.mxErr != 0)
            {
                var pOvflPage = new PgHdr();
                
                if (iPage < 1)
                {
                    checkAppendMsg(pCheck, zContext, "%d of %d pages missing from overflow list starting at %d", N + 1, expected, iFirst);
                    break;
                }
                if (checkRef(pCheck, (uint)iPage, zContext) != 0)
                    break;
                byte[] pOvflData; 
                if (Pager.sqlite3PagerGet(pCheck.pPager, (Pgno)iPage, ref pOvflPage) != 0)
                {
                    checkAppendMsg(pCheck, zContext, "failed to get page %d", iPage);
                    break;
                }
                pOvflData = Pager.sqlite3PagerGetData(pOvflPage);
                if (isFreeList != 0)
                {
                    int n = (int)ConvertEx.sqlite3Get4byte(pOvflData, 4);
#if !SQLITE_OMIT_AUTOVACUUM
                    if (pCheck.pBt.autoVacuum)
                        checkPtrmap(pCheck, (uint)iPage, PTRMAP_FREEPAGE, 0, zContext);
#endif
                    if (n > (int)pCheck.pBt.usableSize / 4 - 2)
                    {
                        checkAppendMsg(pCheck, zContext, "freelist leaf count too big on page %d", iPage);
                        N--;
                    }
                    else
                    {
                        for (i = 0; i < n; i++)
                        {
                            Pgno iFreePage = ConvertEx.sqlite3Get4byte(pOvflData, 8 + i * 4);
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
                        i = (int)ConvertEx.sqlite3Get4byte(pOvflData);
                        checkPtrmap(pCheck, (uint)i, PTRMAP_OVERFLOW2, (uint)iPage, zContext);
                    }
                }
#endif
                iPage = (int)ConvertEx.sqlite3Get4byte(pOvflData);
                Pager.sqlite3PagerUnref(pOvflPage);
            }
        }

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
#endif
    }
}
