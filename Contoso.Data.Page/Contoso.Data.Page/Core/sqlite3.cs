using Contoso.Sys;
using System;
namespace Contoso.Core
{
    public class sqlite3
    {
        public class BusyHandler
        {
            public Func<object, int, int> xFunc;  // The busy callback
            public object pArg;                   // First arg to busy callback
            public int nBusy;                     // Incremented with each busy call
        }

        [Flags]
        public enum SQLITE
        {
            VdbeTrace = 0x00000100,
            InternChanges = 0x00000200,
            FullColNames = 0x00000400,
            ShortColNames = 0x00000800,
            CountRows = 0x00001000,
            NullCallback = 0x00002000,
            SqlTrace = 0x00004000,
            VdbeListing = 0x00008000,
            WriteSchema = 0x00010000,
            NoReadlock = 0x00020000,
            IgnoreChecks = 0x00040000,
            ReadUncommitted = 0x0080000,
            LegacyFileFmt = 0x00100000,
            FullFSync = 0x00200000,
            CkptFullFSync = 0x00400000,
            RecoveryMode = 0x00800000,
            ReverseOrder = 0x01000000,
            RecTriggers = 0x02000000,
            ForeignKeys = 0x04000000,
            AutoIndex = 0x08000000,
            PreferBuiltin = 0x10000000,
            LoadExtension = 0x20000000,
            EnableTrigger = 0x40000000,
        }

        public sqlite3_mutex mutex { get; set; }
        public SQLITE flags { get; set; }
        public BusyHandler busyHandler { get; set; }
        public int nSavepoint { get; set; } // Number of non-transaction savepoints
        public int activeVdbeCnt { get; set; }

        public int sqlite3InvokeBusyHandler()
        {
            if (Check.NEVER(busyHandler == null) || busyHandler.xFunc == null || busyHandler.nBusy < 0)
                return 0;
            var rc = busyHandler.xFunc(busyHandler.pArg, busyHandler.nBusy);
            if (rc == 0)
                busyHandler.nBusy = -1;
            else
                busyHandler.nBusy++;
            return rc;
        }

        // HOOKS
#if SQLITE_ENABLE_UNLOCK_NOTIFY
void sqlite3ConnectionBlocked(sqlite3 *, sqlite3 );
void sqlite3ConnectionUnlocked(sqlite3 db);
void sqlite3ConnectionClosed(sqlite3 db);
#else
        internal static void sqlite3ConnectionBlocked(sqlite3 x, sqlite3 y) { }
        //internal static void sqlite3ConnectionUnlocked(sqlite3 x) { }
        //internal static void sqlite3ConnectionClosed(sqlite3 x) { }
#endif

        internal bool sqlite3TempInMemory()
        {
            return true;
            //if (SQLITE_TEMP_STORE == 1) return (temp_store == 2);
            //if (SQLITE_TEMP_STORE == 2) return (temp_store != 1);
            //if (SQLITE_TEMP_STORE == 3) return true;
            //if (SQLITE_TEMP_STORE < 1 || SQLITE_TEMP_STORE > 3) return false;
            //return false;
        }
    }
}
