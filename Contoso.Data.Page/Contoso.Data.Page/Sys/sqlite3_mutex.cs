namespace Contoso.Sys
{
    public class sqlite3_mutex
    {
        public object mutex;    /* Mutex controlling the lock */
        public int id;          /* Mutex type */
        public int nRef;        /* Number of enterances */
        public int owner;       /* Thread holding this mutex */
#if DEBUG
        public int trace;       /* True to trace changes */
#endif

        public sqlite3_mutex()
        {
            mutex = new object();
        }

        public sqlite3_mutex(sqlite3_mutex mutex, int id, int nRef, int owner
#if DEBUG
, int trace
#endif
)
        {
            this.mutex = mutex;
            this.id = id;
            this.nRef = nRef;
            this.owner = owner;
#if DEBUG
            this.trace = 0;
#endif
        }
    }
}
