namespace Contoso.Threading
{
    public class MutexEx
    {
        public static bool WantsCoreMutex { get; set; }

        private static int mutexIsInit = 0;
        public static bool SQLITE_THREADSAFE;

        public enum MUTEX
        {
            FAST = 0,
            RECURSIVE = 1,
            STATIC_MASTER = 2,
            STATIC_MEM = 3,  // sqlite3_malloc()
            STATIC_MEM2 = 4,  // NOT USED
            STATIC_OPEN = 4,  // sqlite3BtreeOpen()
            STATIC_PRNG = 5,  // sqlite3_random()
            STATIC_LRU = 6,   // lru page list
            STATIC_LRU2 = 7,  // NOT USED
            STATIC_PMEM = 7, // sqlite3PageMalloc()
        }

        public static sqlite3_mutex sqlite3_mutex_alloc(MUTEX id)
        {
            //#if !SQLITE_OMIT_AUTOINIT
            //            if (sqlite3_initialize() != 0) return null;
            //#endif
            //            return sqlite3GlobalConfig.mutex.xMutexAlloc(id);
            return null;
        }

        public static sqlite3_mutex sqlite3MutexAlloc(MUTEX id)
        {
            //if (!sqlite3GlobalConfig.bCoreMutex)
            //    return null;
            //Debug.Assert(mutexIsInit != 0);
            //return sqlite3GlobalConfig.mutex.xMutexAlloc(id);
            return null;
        }

        public static void sqlite3_mutex_enter(sqlite3_mutex sqlite3_mutex) { }
        public static void sqlite3_mutex_leave(sqlite3_mutex sqlite3_mutex) { }
        public static bool Held(sqlite3_mutex sqlite3_mutex) { return true; }
        public static bool sqlite3_mutex_notheld(sqlite3_mutex sqlite3_mutex) { return true; }
        public static void sqlite3_mutex_free(sqlite3_mutex mutex) { }
    }
}
