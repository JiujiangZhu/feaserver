using System;
using System.Diagnostics;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public partial class Pager
    {
        // was:sqlite3PagerGet,sqlite3PagerAcquire
        public RC Get(Pgno pgno, ref DbPage ppPage) { return Get(pgno, ref ppPage, 0); }
        public RC Get(Pgno pgno, ref DbPage ppPage, byte noContent)
        {
            Debug.Assert(eState >= PAGER.READER);
            Debug.Assert(assert_pager_state());
            if (pgno == 0)
                return SysEx.SQLITE_CORRUPT_BKPT();
            // If the pager is in the error state, return an error immediately.  Otherwise, request the page from the PCache layer.
            var rc = (errCode != RC.OK ? errCode : pPCache.FetchPage(pgno, 1, ref ppPage));
            PgHdr pPg = null;
            if (rc != RC.OK)
            {
                // Either the call to sqlite3PcacheFetch() returned an error or the pager was already in the error-state when this function was called.
                // Set pPg to 0 and jump to the exception handler.  */
                pPg = null;
                goto pager_get_err;
            }
            Debug.Assert((ppPage).ID == pgno);
            Debug.Assert((ppPage).Pager == this || (ppPage).Pager == null);
            if ((ppPage).Pager != null && 0 == noContent)
            {
                // In this case the pcache already contains an initialized copy of the page. Return without further ado.
                Debug.Assert(pgno <= PAGER_MAX_PGNO && pgno != PAGER_MJ_PGNO(this));
                return RC.OK;
            }
            else
            {
                // The pager cache has created a new page. Its content needs to be initialized.
                pPg = ppPage;
                pPg.Pager = this;
                pPg.Extra = _memPageBuilder;
                // The maximum page number is 2^31. Return SQLITE_CORRUPT if a page number greater than this, or the unused locking-page, is requested.
                if (pgno > PAGER_MAX_PGNO || pgno == PAGER_MJ_PGNO(this))
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto pager_get_err;
                }
                if (
#if SQLITE_OMIT_MEMORYDB
1==MEMDB
#else
memDb != 0
#endif
 || dbSize < pgno || noContent != 0 || !fd.IsOpen)
                {
                    if (pgno > mxPgno)
                    {
                        rc = RC.FULL;
                        goto pager_get_err;
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
                    Array.Clear(pPg.Data, 0, pageSize);
                    SysEx.IOTRACE("ZERO {0:x} {1}\n", this.GetHashCode(), pgno);
                }
                else
                {
                    Debug.Assert(pPg.Pager == this);
                    rc = readDbPage(pPg);
                    if (rc != RC.OK)
                        goto pager_get_err;
                }
                pager_set_pagehash(pPg);
            }
            return RC.OK;
        pager_get_err:
            Debug.Assert(rc != RC.OK);
            if (pPg != null)
                PCache.DropPage(pPg);
            pagerUnlockIfUnused();
            ppPage = null;
            return rc;
        }

        // was:sqlite3PagerLookup
        public DbPage Lookup(Pgno pgno)
        {
            PgHdr pPg = null;
            Debug.Assert(pgno != 0);
            Debug.Assert(pPCache != null);
            Debug.Assert(eState >= PAGER.READER && eState != PAGER.ERROR);
            pPCache.FetchPage(pgno, 0, ref pPg);
            return pPg;
        }

        // was:sqlite3PagerUnref
        public static void Unref(DbPage pPg)
        {
            if (pPg != null)
            {
                var pPager = pPg.Pager;
                PCache.ReleasePage(pPg);
                pPager.pagerUnlockIfUnused();
            }
        }
    }
}
