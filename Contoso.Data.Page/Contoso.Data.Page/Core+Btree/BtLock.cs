using Pgno = System.UInt32;

namespace Contoso.Core
{
    public class BtLock
    {
        public enum ELOCK
        {
            READ_LOCK = 1,
            WRITE_LOCK = 2,
        }

        public Btree pBtree;        // Btree handle holding this lock
        public Pgno iTable;         // Root page of table
        public ELOCK eLock;         // READ_LOCK or WRITE_LOCK
        public BtLock pNext;        // Next in BtShared.pLock list
    }
}
