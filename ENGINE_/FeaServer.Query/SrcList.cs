using FeaServer.Core;
namespace FeaServer
{
    public class SrcList
    {
        public struct Item
        {
            public string Database;     /* Name of database holding this table */
            public string Name;         /* Name of the table */
            public string Alias;        /* The "B" part of a "A AS B" phrase.  zName is the "A" */
            public Table Tab;           /* An SQL table corresponding to zName */
            public Select Select;       /* A SELECT statement used in place of a table name */
            public int addrFillSub;     /* Address of subroutine to manifest a subquery */
            public int regReturn;       /* Register holding return address of addrFillSub */
            public byte jointype;       /* Type of join between this able and the previous */
            public bool notIndexed;     /* True if there is a NOT INDEXED clause */
            public bool isCorrelated;   /* True if sub-query is correlated */
            public byte SelectId;       /* If Select != null, the id of the sub-select in EQP */
            public int Cursor;          /* The VDBE cursor number used to access this table */
            public Expr On;             /* The ON clause of a join */
            public IdList Using;        /* The USING clause of a join */
            public uint colUsed;        /* Bit N (1<<N) set if column N of pTab is used */
            //public string Index;      /* Identifier from "INDEXED BY <zIndex>" clause */
            //public Index zIndex;      /* Index structure corresponding to zIndex, if any */
        }
        public short Srcs;              /* Number of tables or subqueries in the FROM clause */
        public short Allocs;            /* Number of entries allocated in a[] below */
        public Item[] A = new Item[1];  /* One entry for each identifier on the list */
    }
}
