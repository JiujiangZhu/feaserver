using System;
using Pgno = System.UInt32;
using System.Diagnostics;
using Contoso.Sys;
using System.Text;
namespace Contoso.Core
{
    public partial class Pager
    {
        public enum SAVEPOINT
        {
            BEGIN = 0,
            RELEASE = 1,
            ROLLBACK = 2,
        }

        internal static SQLITE pagerPlaybackSavepoint(Pager pPager, PagerSavepoint pSavepoint)
        {
            Debug.Assert(pPager.eState != PAGER.ERROR);
            Debug.Assert(pPager.eState >= PAGER.WRITER_LOCKED);
            // Allocate a bitvec to use to store the set of pages rolled back
            Bitvec pDone = null;     // Bitvec to ensure pages played back only once
            if (pSavepoint != null)
                pDone = Bitvec.sqlite3BitvecCreate(pSavepoint.nOrig);
            // Set the database size back to the value it was before the savepoint being reverted was opened.
            pPager.dbSize = pSavepoint != null ? pSavepoint.nOrig : pPager.dbOrigSize;
            pPager.changeCountDone = pPager.tempFile;
            if (!pSavepoint && pPager.pagerUseWal())
                return pPager.pagerRollbackWal();
            // Use pPager.journalOff as the effective size of the main rollback journal.  The actual file might be larger than this in
            // PAGER_JOURNALMODE_TRUNCATE or PAGER_JOURNALMODE_PERSIST.  But anything past pPager.journalOff is off-limits to us.
            var szJ = pPager.journalOff; // Effective size of the main journal
            Debug.Assert(!pPager.pagerUseWal() || szJ == 0);
            // Begin by rolling back records from the main journal starting at PagerSavepoint.iOffset and continuing to the next journal header.
            // There might be records in the main journal that have a page number greater than the current database size (pPager.dbSize) but those
            // will be skipped automatically.  Pages are added to pDone as they are played back.
            long iHdrOff;             // End of first segment of main-journal records
            var rc = SQLITE.OK;      // Return code
            if (pSavepoint != null && !pPager.pagerUseWal())
            {
                iHdrOff = (pSavepoint.iHdrOffset != 0 ? pSavepoint.iHdrOffset : szJ);
                pPager.journalOff = pSavepoint.iOffset;
                while (rc == SQLITE.OK && pPager.journalOff < iHdrOff)
                    rc = pager_playback_one_page(pPager, ref pPager.journalOff, pDone, 1, 1);
                Debug.Assert(rc != SQLITE.DONE);
            }
            else
                pPager.journalOff = 0;
            // Continue rolling back records out of the main journal starting at the first journal header seen and continuing until the effective end
            // of the main journal file.  Continue to skip out-of-range pages and continue adding pages rolled back to pDone.
            while (rc == SQLITE.OK && pPager.journalOff < szJ)
            {
                uint nJRec;         // Number of Journal Records
                uint dummy;
                rc = readJournalHdr(pPager, 0, szJ, out nJRec, out dummy);
                Debug.Assert(rc != SQLITE.DONE);
                // The "pPager.journalHdr+JOURNAL_HDR_SZ(pPager)==pPager.journalOff" test is related to ticket #2565.  See the discussion in the
                // pager_playback() function for additional information.
                if (nJRec == 0 && pPager.journalHdr + JOURNAL_HDR_SZ(pPager) >= pPager.journalOff)
                    nJRec = (uint)((szJ - pPager.journalOff) / JOURNAL_PG_SZ(pPager));
                for (uint ii = 0; rc == SQLITE.OK && ii < nJRec && pPager.journalOff < szJ; ii++)
                    rc = pager_playback_one_page(pPager, ref pPager.journalOff, pDone, 1, 1);
                Debug.Assert(rc != SQLITE.DONE);
            }
            Debug.Assert(rc != SQLITE.OK || pPager.journalOff >= szJ);
            // Finally,  rollback pages from the sub-journal.  Page that were previously rolled back out of the main journal (and are hence in pDone)
            // will be skipped.  Out-of-range pages are also skipped.
            if (pSavepoint != null)
            {
                long offset = pSavepoint.iSubRec * (4 + pPager.pageSize);
                if (pPager.pagerUseWal())
                    rc = Wal.sqlite3WalSavepointUndo(pPager.pWal, pSavepoint.aWalData);
                for (var ii = pSavepoint.iSubRec; rc == SQLITE.OK && ii < pPager.nSubRec; ii++)
                {
                    Debug.Assert(offset == ii * (4 + pPager.pageSize));
                    rc = pager_playback_one_page(pPager, ref offset, pDone, 0, 1);
                }
                Debug.Assert(rc != SQLITE.DONE);
            }
            Bitvec.sqlite3BitvecDestroy(ref pDone);
            if (rc == SQLITE.OK)
                pPager.journalOff = (int)szJ;
            return rc;
        }

        internal static void releaseAllSavepoints(Pager pPager)
        {
            for (var ii = 0; ii < pPager.nSavepoint; ii++)
                Bitvec.sqlite3BitvecDestroy(ref pPager.aSavepoint[ii].pInSavepoint);
            if (!pPager.exclusiveMode || pPager.sjfd is MemJournalFile)
                FileEx.sqlite3OsClose(pPager.sjfd);
            pPager.aSavepoint = null;
            pPager.nSavepoint = 0;
            pPager.nSubRec = 0;
        }

        internal static SQLITE addToSavepointBitvecs(Pager pPager, Pgno pgno)
        {
            var rc = SQLITE.OK;
            for (var ii = 0; ii < pPager.nSavepoint; ii++)
            {
                var p = pPager.aSavepoint[ii];
                if (pgno <= p.nOrig)
                {
                    rc |= p.pInSavepoint.sqlite3BitvecSet(pgno);
                    Debug.Assert(rc == SQLITE.OK || rc == SQLITE.NOMEM);
                }
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerOpenSavepoint(Pager pPager, int nSavepoint)
        {
            var rc = SQLITE.OK;
            var nCurrent = pPager.nSavepoint;        // Current number of savepoints 
            Debug.Assert(pPager.eState >= PAGER.WRITER_LOCKED);
            Debug.Assert(assert_pager_state(pPager));
            if (nSavepoint > nCurrent && pPager.useJournal != 0)
            {
                // Grow the Pager.aSavepoint array using realloc(). Return SQLITE_NOMEM if the allocation fails. Otherwise, zero the new portion in case a 
                // malloc failure occurs while populating it in the for(...) loop below.
                Array.Resize(ref pPager.aSavepoint, nSavepoint);
                var aNew = pPager.aSavepoint; // New Pager.aSavepoint array
                // Populate the PagerSavepoint structures just allocated.
                for (var ii = nCurrent; ii < nSavepoint; ii++)
                {
                    aNew[ii] = new PagerSavepoint();
                    aNew[ii].nOrig = pPager.dbSize;
                    aNew[ii].iOffset = (pPager.jfd.isOpen && pPager.journalOff > 0 ? pPager.journalOff : (int)JOURNAL_HDR_SZ(pPager));
                    aNew[ii].iSubRec = pPager.nSubRec;
                    aNew[ii].pInSavepoint = Bitvec.sqlite3BitvecCreate(pPager.dbSize);
                    if (pagerUseWal(pPager))
                        Wal.sqlite3WalSavepoint(pPager.pWal, aNew[ii].aWalData);
                    pPager.nSavepoint = ii + 1;
                }
                Debug.Assert(pPager.nSavepoint == nSavepoint);
                assertTruncateConstraint(pPager);
            }
            return rc;
        }

        internal static SQLITE sqlite3PagerSavepoint(Pager pPager, SAVEPOINT op, int iSavepoint)
        {
            var rc = pPager.errCode;
            Debug.Assert(op == SAVEPOINT.RELEASE || op == SAVEPOINT.ROLLBACK);
            Debug.Assert(iSavepoint >= 0 || op == SAVEPOINT.ROLLBACK);
            if (rc == SQLITE.OK && iSavepoint < pPager.nSavepoint)
            {
                // Figure out how many savepoints will still be active after this operation. Store this value in nNew. Then free resources associated
                // with any savepoints that are destroyed by this operation.
                var nNew = iSavepoint + ((op == SAVEPOINT.RELEASE) ? 0 : 1); // Number of remaining savepoints after this op.
                for (var ii = nNew; ii < pPager.nSavepoint; ii++)
                    Bitvec.sqlite3BitvecDestroy(ref pPager.aSavepoint[ii].pInSavepoint);
                pPager.nSavepoint = nNew;
                // If this is a release of the outermost savepoint, truncate the sub-journal to zero bytes in size.
                if (op == SAVEPOINT.RELEASE)
                    if (nNew == 0 && pPager.sjfd.isOpen)
                    {
                        // Only truncate if it is an in-memory sub-journal.
                        if (pPager.sjfd is MemJournalFile)
                        {
                            rc = FileEx.sqlite3OsTruncate(pPager.sjfd, 0);
                            Debug.Assert(rc == SQLITE.OK);
                        }
                        pPager.nSubRec = 0;
                    }
                    // Else this is a rollback operation, playback the specified savepoint. If this is a temp-file, it is possible that the journal file has
                    // not yet been opened. In this case there have been no changes to the database file, so the playback operation can be skipped.
                    else if (pagerUseWal(pPager) || pPager.jfd.isOpen)
                    {
                        var pSavepoint = (nNew == 0 ? (PagerSavepoint)null : pPager.aSavepoint[nNew - 1]);
                        rc = pagerPlaybackSavepoint(pPager, pSavepoint);
                        Debug.Assert(rc != SQLITE.DONE);
                    }
            }
            return rc;
        }
    }
}
