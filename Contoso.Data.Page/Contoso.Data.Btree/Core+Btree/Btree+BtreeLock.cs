using Pgno = System.UInt32;

namespace Contoso.Core
{
    public partial class Btree
    {
        // was:BtLock
        public class BtreeLock
        {
            public enum LOCK
            {
                READ = 1,
                WRITE = 2,
            }

            public Btree Tree;          // Btree handle holding this lock
            public Pgno TableID;    // Root page of table
            public LOCK Lock;         // READ_LOCK or WRITE_LOCK
            public BtreeLock Next;      // Next in BtShared.pLock list
        }
    }
}
