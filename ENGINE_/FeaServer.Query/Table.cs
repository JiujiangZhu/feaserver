using FeaServer.Core;
namespace FeaServer
{
    public class Table
    {
        public int PKey;            /* If not negative, use aCol[iPKey] as the primary key */
        public int Cols;            /* Number of columns in this table */
        public Column aCol;         /* Information about each column */
        public int tnum;            /* Root BTree node for this table (see note above) */
        //public tRowcnt RowEsts;     /* Estimated rows in table - from sqlite_stat1 table */
        public Select Select;       /* NULL for tables.  Points to definition if a view. */
        public ushort Refs;         /* Number of pointers to this Table */
        public byte tabFlags;       /* Mask of TF_* values */
        public byte keyConf;        /* What to do in case of uniqueness conflict on iPKey */
        public string ColAff;       /* String defining the affinity of each column */
        public Expr Check;          /* The AND of all CHECK constraints */
        public int addColOffset;    /* Offset in CREATE TABLE stmt to add a new column */
        //#ifndef SQLITE_OMIT_VIRTUALTABLE
        //  VTable *pVTable;        /* List of VTable objects. */
        //  int nModuleArg;         /* Number of arguments to the module */
        //  char **azModuleArg;     /* Text of all module args. [0] is module name */
        //#endif
        public Trigger Trigger;     /* List of triggers stored in pSchema */
        public Schema Schema;       /* Schema that contains this table */
        public Table NextZombie;    /* Next on the Parse.pZombieTab list */
    }
}
