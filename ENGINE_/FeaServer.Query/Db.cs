using FeaServer.Core;
namespace FeaServer
{
    /// <summary>
    /// Each database file to be accessed by the system is an instance of the following structure.  There are normally two of these structures
    /// in the sqlite.aDb[] array.  aDb[0] is the main database file and aDb[1] is the database file used to hold temporary tables.  Additional databases may be attached.
    /// </summary>
    public class Db
    {
        public string Name;         /* Name of this database */
        //public Btree Bt;            /* The B*Tree structure for this database file */
        public byte inTrans;        /* 0: not writable.  1: Transaction.  2: Checkpoint */
        public byte safety_level;   /* How aggressive at syncing data to disk */
        public Schema Schema;       /* Pointer to database schema (possibly shared) */
    }
}
