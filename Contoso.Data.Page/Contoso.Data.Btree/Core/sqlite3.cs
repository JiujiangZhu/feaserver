using System;
namespace Contoso.Core
{
    public class sqlite3 : sqlite3b
    {
        private const int SQLITE_MAX_ATTACHED = 10;

        public class Db
        {
            public string Name;         // Name of this database
            public Btree Tree;          // The B Tree structure for this database file
            public byte InTransaction;  // 0: not writable.  1: Transaction.  2: Checkpoint
            public byte SafetyLevel;    // How aggressive at syncing data to disk 
            public ISchema Schema;      // Pointer to database schema (possibly shared)
        }

        public Db[] AllocDBs = new Db[SQLITE_MAX_ATTACHED];
        public int DBs { get; set; }
    }
}
