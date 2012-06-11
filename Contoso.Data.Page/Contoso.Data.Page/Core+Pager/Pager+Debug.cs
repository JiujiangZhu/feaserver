using System.Diagnostics;
using Contoso.Sys;
using VFSLOCK = Contoso.Sys.VirtualFile.LOCK;
using System;
namespace Contoso.Core
{
    public partial class Pager
    {
#if TRACE
        private static bool sqlite3PagerTrace = true;  // True to enable tracing
        private static void PAGERTRACE(string x, params object[] args) { if (sqlite3PagerTrace)Console.WriteLine(string.Format(x, args)); }
#else
        private static void PAGERTRACE(string x, params object[] args) { }
#endif
        private static int PAGERID(Pager p) { return p.GetHashCode(); }
        private static int FILEHANDLEID(VirtualFile fd) { return fd.GetHashCode(); }

#if DEBUG
        internal bool assert_pager_state()
        {
            // Regardless of the current state, a temp-file connection always behaves as if it has an exclusive lock on the database file. It never updates
            // the change-counter field, so the changeCountDone flag is always set.
            Debug.Assert(!tempFile || eLock == VFSLOCK.EXCLUSIVE);
            Debug.Assert(!tempFile || changeCountDone);
            // If the useJournal flag is clear, the journal-mode must be "OFF". And if the journal-mode is "OFF", the journal file must not be open.
            Debug.Assert(journalMode == JOURNALMODE.OFF || useJournal != 0);
            Debug.Assert(journalMode != JOURNALMODE.OFF || !jfd.isOpen);
            // Check that MEMDB implies noSync. And an in-memory journal. Since  this means an in-memory pager performs no IO at all, it cannot encounter 
            // either SQLITE_IOERR or SQLITE_FULL during rollback or while finalizing a journal file. (although the in-memory journal implementation may 
            // return SQLITE_IOERR_NOMEM while the journal file is being written). It is therefore not possible for an in-memory pager to enter the ERROR state.
            if (
#if SQLITE_OMIT_MEMORYDB
0!=MEMDB
#else
0 != memDb
#endif
)
            {
                Debug.Assert(noSync);
                Debug.Assert(journalMode == JOURNALMODE.OFF || journalMode == JOURNALMODE.MEMORY);
                Debug.Assert(eState != PAGER.ERROR && eState != PAGER.OPEN);
                Debug.Assert(!pagerUseWal());
            }
            // If changeCountDone is set, a RESERVED lock or greater must be held on the file.
            Debug.Assert(!changeCountDone || eLock >= VFSLOCK.RESERVED);
            Debug.Assert(eLock != VFSLOCK.PENDING);
            switch (eState)
            {
                case PAGER.OPEN:
                    Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == memDb
#endif
);
                    Debug.Assert(errCode == RC.OK);
                    Debug.Assert(pPCache.sqlite3PcacheRefCount() == 0 || tempFile);
                    break;
                case PAGER.READER:
                    Debug.Assert(errCode == RC.OK);
                    Debug.Assert(eLock != VFSLOCK.UNKNOWN);
                    Debug.Assert(eLock >= VFSLOCK.SHARED || noReadlock != 0);
                    break;
                case PAGER.WRITER_LOCKED:
                    Debug.Assert(eLock != VFSLOCK.UNKNOWN);
                    Debug.Assert(errCode == RC.OK);
                    if (!pagerUseWal())
                        Debug.Assert(eLock >= VFSLOCK.RESERVED);
                    Debug.Assert(dbSize == dbOrigSize);
                    Debug.Assert(dbOrigSize == dbFileSize);
                    Debug.Assert(dbOrigSize == dbHintSize);
                    Debug.Assert(setMaster == 0);
                    break;
                case PAGER.WRITER_CACHEMOD:
                    Debug.Assert(eLock != VFSLOCK.UNKNOWN);
                    Debug.Assert(errCode == RC.OK);
                    if (!pagerUseWal())
                    {
                        // It is possible that if journal_mode=wal here that neither the journal file nor the WAL file are open. This happens during
                        // a rollback transaction that switches from journal_mode=off to journal_mode=wal.
                        Debug.Assert(eLock >= VFSLOCK.RESERVED);
                        Debug.Assert(jfd.isOpen || journalMode == JOURNALMODE.OFF || journalMode == JOURNALMODE.WAL);
                    }
                    Debug.Assert(dbOrigSize == dbFileSize);
                    Debug.Assert(dbOrigSize == dbHintSize);
                    break;
                case PAGER.WRITER_DBMOD:
                    Debug.Assert(eLock == VFSLOCK.EXCLUSIVE);
                    Debug.Assert(errCode == RC.OK);
                    Debug.Assert(!pagerUseWal());
                    Debug.Assert(eLock >= VFSLOCK.EXCLUSIVE);
                    Debug.Assert(jfd.isOpen || journalMode == JOURNALMODE.OFF || journalMode == JOURNALMODE.WAL);
                    Debug.Assert(dbOrigSize <= dbHintSize);
                    break;
                case PAGER.WRITER_FINISHED:
                    Debug.Assert(eLock == VFSLOCK.EXCLUSIVE);
                    Debug.Assert(errCode == RC.OK);
                    Debug.Assert(!pagerUseWal());
                    Debug.Assert(jfd.isOpen || journalMode == JOURNALMODE.OFF || journalMode == JOURNALMODE.WAL);
                    break;
                case PAGER.ERROR:
                    // There must be at least one outstanding reference to the pager if in ERROR state. Otherwise the pager should have already dropped
                    // back to OPEN state.
                    Debug.Assert(errCode != RC.OK);
                    Debug.Assert(pPCache.sqlite3PcacheRefCount() > 0);
                    break;
            }

            return true;
        }
#else
        internal bool assert_pager_state() { return true; }
#endif

#if DEBUG
        // Return a pointer to a human readable string in a static buffer containing the state of the Pager object passed as an argument. This
        // is intended to be used within debuggers. For example, as an alternative to "print *pPager" in gdb:
        // (gdb) printf "%s", print_pager_state(pPager)
        internal string print_pager_state()
        {
            return string.Format(@"
Filename:      {0}
State:         {1} errCode={2}
Lock:          {3}
Locking mode:  locking_mode={4}
Journal mode:  journal_mode={5}
Backing store: tempFile={6} memDb={7} useJournal={8}
Journal:       journalOff={9.11} journalHdr={10.11}
Size:          dbsize={11} dbOrigSize={12} dbFileSize={13}"
          , zFilename
          , eState == PAGER.OPEN ? "OPEN" :
              eState == PAGER.READER ? "READER" :
              eState == PAGER.WRITER_LOCKED ? "WRITER_LOCKED" :
              eState == PAGER.WRITER_CACHEMOD ? "WRITER_CACHEMOD" :
              eState == PAGER.WRITER_DBMOD ? "WRITER_DBMOD" :
              eState == PAGER.WRITER_FINISHED ? "WRITER_FINISHED" :
              eState == PAGER.ERROR ? "ERROR" : "?error?"
          , (int)errCode
          , eLock == VFSLOCK.NO ? "NO_LOCK" :
              eLock == VFSLOCK.RESERVED ? "RESERVED" :
              eLock == VFSLOCK.EXCLUSIVE ? "EXCLUSIVE" :
              eLock == VFSLOCK.SHARED ? "SHARED" :
              eLock == VFSLOCK.UNKNOWN ? "UNKNOWN" : "?error?"
          , exclusiveMode ? "exclusive" : "normal"
          , journalMode == JOURNALMODE.MEMORY ? "memory" :
              journalMode == JOURNALMODE.OFF ? "off" :
              journalMode == JOURNALMODE.DELETE ? "delete" :
              journalMode == JOURNALMODE.PERSIST ? "persist" :
              journalMode == JOURNALMODE.TRUNCATE ? "truncate" :
              journalMode == JOURNALMODE.WAL ? "wal" : "?error?"
          , tempFile ? 1 : 0, (int)memDb, (int)useJournal
          , journalOff, journalHdr
          , (int)dbSize, (int)dbOrigSize, (int)dbFileSize);
        }
#endif

#if DEBUG
        internal static void assertTruncateConstraintCb(PgHdr pPg) { Debug.Assert((pPg.flags & PgHdr.PGHDR.DIRTY) != 0); Debug.Assert(!subjRequiresPage(pPg) || pPg.pgno <= pPg.pPager.dbSize); }
        internal void assertTruncateConstraint() { pPCache.sqlite3PcacheIterateDirty(assertTruncateConstraintCb); }
#else
        internal static void assertTruncateConstraintCb(PgHdr pPg) { }
        internal void assertTruncateConstraint() { }
#endif
    }
}
