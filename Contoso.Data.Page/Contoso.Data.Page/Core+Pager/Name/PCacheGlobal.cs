using Contoso.Sys;
namespace Contoso.Core.Name
{
    public class PCacheGlobal
    {
        static PCacheGlobal pcache = new PCacheGlobal();

        public PGroup grp = new PGroup();   // The global PGroup for mode (2)
        // Variables related to SQLITE_CONFIG_PAGECACHE settings.  The szSlot, nSlot, pStart, pEnd, nReserve, and isInit values are all
        // fixed at sqlite3_initialize() time and do not require mutex protection. The nFreeSlot and pFree values do require mutex protection.
        public bool isInit;                 // True if initialized
        public int szSlot;                  // Size of each free slot
        public int nSlot;                   // The number of pcache slots
        public int nReserve;                // Try to keep nFreeSlot above this
        public object pStart, pEnd;         // Bounds of pagecache malloc range
        // Above requires no mutex.  Use mutex below for variable that follow.
        public sqlite3_mutex mutex;         // Mutex for accessing the following:
        public int nFreeSlot;               // Number of unused pcache slots
        public PgFreeslot pFree;            // Free page blocks
        // The following value requires a mutex to change.  We skip the mutex on reading because (1) most platforms read a 32-bit integer atomically and
        // (2) even if an incorrect value is read, no great harm is done since this is really just an optimization.
        public bool bUnderPressure;         // True if low on PAGECACHE memory
    }
}
