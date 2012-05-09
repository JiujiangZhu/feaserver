namespace Contoso.Core
{
    public class Btree
    {
        public enum TRANS : byte
        {
            NONE = 0,
            READ = 1,
            WRITE = 2,
        }

        public sqlite3 db;        /* The database connection holding this Btree */
        public BtShared pBt;      /* Sharable content of this Btree */
        public TRANS inTrans;        /* TRANS_NONE, TRANS_READ or TRANS_WRITE */
        public bool sharable;     /* True if we can share pBt with another db */
        public bool locked;       /* True if db currently has pBt locked */
        public int wantToLock;    /* Number of nested calls to sqlite3BtreeEnter() */
        public int nBackup;       /* Number of backup operations reading this btree */
        public Btree pNext;       /* List of other sharable Btrees from the same db */
        public Btree pPrev;       /* Back pointer of the same list */
        //#if !SQLITE_OMIT_SHARED_CACHE
        //BtLock lock;              /* Object used to lock page 1 */
        //#endif
    }
}
