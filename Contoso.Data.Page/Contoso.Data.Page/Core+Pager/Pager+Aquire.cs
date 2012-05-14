using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using System.Text;
using DbPage = Contoso.Core.PgHdr;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal SQLITE sqlite3PagerGet(Pgno pgno, ref DbPage ppPage) { return sqlite3PagerAcquire(pgno, ref ppPage, 0); }
        internal SQLITE sqlite3PagerAcquire(Pgno pgno, ref DbPage ppPage, byte noContent)
        {
            Debug.Assert(eState >= PAGER.READER);
            Debug.Assert(assert_pager_state());
            if (pgno == 0)
                return SysEx.SQLITE_CORRUPT_BKPT();
            // If the pager is in the error state, return an error immediately.  Otherwise, request the page from the PCache layer.
            var rc = (errCode != SQLITE.OK ? errCode : pPCache.sqlite3PcacheFetch(pgno, 1, ref ppPage));
            PgHdr pPg = null;
            if (rc != SQLITE.OK)
            {
                // Either the call to sqlite3PcacheFetch() returned an error or the pager was already in the error-state when this function was called.
                // Set pPg to 0 and jump to the exception handler.  */
                pPg = null;
                goto pager_acquire_err;
            }
            Debug.Assert((ppPage).pgno == pgno);
            Debug.Assert((ppPage).pPager == this || (ppPage).pPager == null);
            if ((ppPage).pPager != null && 0 == noContent)
            {
                // In this case the pcache already contains an initialized copy of the page. Return without further ado.
                Debug.Assert(pgno <= PAGER_MAX_PGNO && pgno != PAGER_MJ_PGNO(this));
                return SQLITE.OK;
            }
            else
            {
                // The pager cache has created a new page. Its content needs to be initialized.
                pPg = ppPage;
                pPg.pPager = this;
                pPg.pExtra = new MemPage();
                // The maximum page number is 2^31. Return SQLITE_CORRUPT if a page number greater than this, or the unused locking-page, is requested.
                if (pgno > PAGER_MAX_PGNO || pgno == PAGER_MJ_PGNO(this))
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto pager_acquire_err;
                }
                if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
memDb != 0
#endif
 || dbSize < pgno || noContent != 0 || !fd.isOpen)
                {
                    if (pgno > mxPgno)
                    {
                        rc = SQLITE.FULL;
                        goto pager_acquire_err;
                    }
                    if (noContent != 0)
                    {
                        // Failure to set the bits in the InJournal bit-vectors is benign. It merely means that we might do some extra work to journal a
                        // page that does not need to be journaled.  Nevertheless, be sure to test the case where a malloc error occurs while trying to set
                        // a bit in a bit vector.
                        MallocEx.sqlite3BeginBenignMalloc();
                        if (pgno <= dbOrigSize)
                            pInJournal.sqlite3BitvecSet(pgno);
                        addToSavepointBitvecs(pgno);
                        MallocEx.sqlite3EndBenignMalloc();
                    }
                    Array.Clear(pPg.pData, 0, pageSize);
                    SysEx.IOTRACE("ZERO %p %d\n", this, pgno);
                }
                else
                {
                    Debug.Assert(pPg.pPager == this);
                    rc = readDbPage(pPg);
                    if (rc != SQLITE.OK)
                        goto pager_acquire_err;
                }
                pager_set_pagehash(pPg);
            }
            return SQLITE.OK;
        pager_acquire_err:
            Debug.Assert(rc != SQLITE.OK);
            if (pPg != null)
                PCache.sqlite3PcacheDrop(pPg);
            pagerUnlockIfUnused(this);
            ppPage = null;
            return rc;
        }

        internal DbPage sqlite3PagerLookup(Pgno pgno)
        {
            PgHdr pPg = null;
            Debug.Assert(pgno != 0);
            Debug.Assert(pPCache != null);
            Debug.Assert(eState >= PAGER.READER && eState != PAGER.ERROR);
            pPCache.sqlite3PcacheFetch(pgno, 0, ref pPg);
            return pPg;
        }

        internal static void sqlite3PagerUnref(DbPage pPg)
        {
            if (pPg != null)
            {
                var pPager = pPg.pPager;
                PCache.sqlite3PcacheRelease(pPg);
                pagerUnlockIfUnused(pPager);
            }
        }
    }
}
