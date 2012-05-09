using System.Diagnostics;
using Contoso.Sys;
namespace Contoso.Core
{
    public partial class Pager
    {
        internal static void TRACE(string T, params object[] ap)
        {
        }

#if DEBUG
        internal static bool assert_pager_state(Pager p)
        {
            Pager pPager = p;
            // Regardless of the current state, a temp-file connection always behaves as if it has an exclusive lock on the database file. It never updates
            // the change-counter field, so the changeCountDone flag is always set.
            Debug.Assert(p.tempFile == false || p.eLock == LOCK.EXCLUSIVE);
            Debug.Assert(p.tempFile == false || pPager.changeCountDone);
            // If the useJournal flag is clear, the journal-mode must be "OFF". And if the journal-mode is "OFF", the journal file must not be open.
            Debug.Assert(p.journalMode == JOURNALMODE.OFF || p.useJournal != 0);
            Debug.Assert(p.journalMode != JOURNALMODE.OFF || !p.jfd.isOpen);
            // Check that MEMDB implies noSync. And an in-memory journal. Since  this means an in-memory pager performs no IO at all, it cannot encounter 
            // either SQLITE_IOERR or SQLITE_FULL during rollback or while finalizing a journal file. (although the in-memory journal implementation may 
            // return SQLITE_IOERR_NOMEM while the journal file is being written). It is therefore not possible for an in-memory pager to enter the ERROR state.
            if (
#if SQLITE_OMIT_MEMORYDB
0!=MEMDB
#else
0 != pPager.memDb
#endif
)
            {
                Debug.Assert(p.noSync);
                Debug.Assert(p.journalMode == JOURNALMODE.OFF || p.journalMode == JOURNALMODE.MEMORY);
                Debug.Assert(p.eState != PAGER.ERROR && p.eState != PAGER.OPEN);
                Debug.Assert(pagerUseWal(p) == false);
            }
            // If changeCountDone is set, a RESERVED lock or greater must be held on the file.
            Debug.Assert(pPager.changeCountDone == false || pPager.eLock >= LOCK.RESERVED);
            Debug.Assert(p.eLock != LOCK.PENDING);
            switch (p.eState)
            {
                case PAGER.OPEN:
                    Debug.Assert(
#if SQLITE_OMIT_MEMORYDB
0==MEMDB
#else
0 == pPager.memDb
#endif
);
                    Debug.Assert(pPager.errCode == SQLITE.OK);
                    Debug.Assert(sqlite3PcacheRefCount(pPager.pPCache) == 0 || pPager.tempFile);
                    break;
                case PAGER.READER:
                    Debug.Assert(pPager.errCode == SQLITE.OK);
                    Debug.Assert(p.eLock != LOCK.UNKNOWN);
                    Debug.Assert(p.eLock >= LOCK.SHARED || p.noReadlock != 0);
                    break;
                case PAGER.WRITER_LOCKED:
                    Debug.Assert(p.eLock != LOCK.UNKNOWN);
                    Debug.Assert(pPager.errCode == SQLITE.OK);
                    if (!pagerUseWal(pPager))
                        Debug.Assert(p.eLock >= LOCK.RESERVED);
                    Debug.Assert(pPager.dbSize == pPager.dbOrigSize);
                    Debug.Assert(pPager.dbOrigSize == pPager.dbFileSize);
                    Debug.Assert(pPager.dbOrigSize == pPager.dbHintSize);
                    Debug.Assert(pPager.setMaster == 0);
                    break;
                case PAGER.WRITER_CACHEMOD:
                    Debug.Assert(p.eLock != LOCK.UNKNOWN);
                    Debug.Assert(pPager.errCode == SQLITE.OK);
                    if (!pagerUseWal(pPager))
                    {
                        // It is possible that if journal_mode=wal here that neither the journal file nor the WAL file are open. This happens during
                        // a rollback transaction that switches from journal_mode=off to journal_mode=wal.
                        Debug.Assert(p.eLock >= LOCK.RESERVED);
                        Debug.Assert(p.jfd.isOpen || p.journalMode == JOURNALMODE.OFF || p.journalMode == JOURNALMODE.WAL);
                    }
                    Debug.Assert(pPager.dbOrigSize == pPager.dbFileSize);
                    Debug.Assert(pPager.dbOrigSize == pPager.dbHintSize);
                    break;
                case PAGER.WRITER_DBMOD:
                    Debug.Assert(p.eLock == LOCK.EXCLUSIVE);
                    Debug.Assert(pPager.errCode == SQLITE.OK);
                    Debug.Assert(!pagerUseWal(pPager));
                    Debug.Assert(p.eLock >= LOCK.EXCLUSIVE);
                    Debug.Assert(p.jfd.isOpen || p.journalMode == JOURNALMODE.OFF || p.journalMode == JOURNALMODE.WAL);
                    Debug.Assert(pPager.dbOrigSize <= pPager.dbHintSize);
                    break;
                case PAGER.WRITER_FINISHED:
                    Debug.Assert(p.eLock == LOCK.EXCLUSIVE);
                    Debug.Assert(pPager.errCode == SQLITE.OK);
                    Debug.Assert(!pagerUseWal(pPager));
                    Debug.Assert(p.jfd.isOpen || p.journalMode == JOURNALMODE.OFF || p.journalMode == JOURNALMODE.WAL);
                    break;
                case PAGER.ERROR:
                    // There must be at least one outstanding reference to the pager if in ERROR state. Otherwise the pager should have already dropped
                    // back to OPEN state.
                    Debug.Assert(pPager.errCode != SQLITE.OK);
                    Debug.Assert(sqlite3PcacheRefCount(pPager.pPCache) > 0);
                    break;
            }

            return true;
        }
#else
        internal static bool assert_pager_state(Pager pPager) { return true; }
#endif

#if DEBUG
        // Return a pointer to a human readable string in a static buffer containing the state of the Pager object passed as an argument. This
        // is intended to be used within debuggers. For example, as an alternative to "print *pPager" in gdb:
        // (gdb) printf "%s", print_pager_state(pPager)
        internal static string print_pager_state(Pager p)
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
          , p.zFilename
          , p.eState == PAGER.OPEN ? "OPEN" :
              p.eState == PAGER.READER ? "READER" :
              p.eState == PAGER.WRITER_LOCKED ? "WRITER_LOCKED" :
              p.eState == PAGER.WRITER_CACHEMOD ? "WRITER_CACHEMOD" :
              p.eState == PAGER.WRITER_DBMOD ? "WRITER_DBMOD" :
              p.eState == PAGER.WRITER_FINISHED ? "WRITER_FINISHED" :
              p.eState == PAGER.ERROR ? "ERROR" : "?error?"
          , (int)p.errCode
          , p.eLock == LOCK.NO ? "NO_LOCK" :
              p.eLock == LOCK.RESERVED ? "RESERVED" :
              p.eLock == LOCK.EXCLUSIVE ? "EXCLUSIVE" :
              p.eLock == LOCK.SHARED ? "SHARED" :
              p.eLock == LOCK.UNKNOWN ? "UNKNOWN" : "?error?"
          , p.exclusiveMode ? "exclusive" : "normal"
          , p.journalMode == JOURNALMODE.MEMORY ? "memory" :
              p.journalMode == JOURNALMODE.OFF ? "off" :
              p.journalMode == JOURNALMODE.DELETE ? "delete" :
              p.journalMode == JOURNALMODE.PERSIST ? "persist" :
              p.journalMode == JOURNALMODE.TRUNCATE ? "truncate" :
              p.journalMode == JOURNALMODE.WAL ? "wal" : "?error?"
          , p.tempFile ? 1 : 0, (int)p.memDb, (int)p.useJournal
          , p.journalOff, p.journalHdr
          , (int)p.dbSize, (int)p.dbOrigSize, (int)p.dbFileSize);
        }
#endif
    }
}
