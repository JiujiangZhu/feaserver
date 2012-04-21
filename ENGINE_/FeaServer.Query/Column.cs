using FeaServer.Core;
namespace FeaServer
{
    public class Column
    {
        public string Name;     /* Name of this column */
        public Expr pDflt;      /* Default value of this column */
        public string Dflt;     /* Original text of the default value */
        public string Type;     /* Data type for this column */
        public string zColl;    /* Collating sequence.  If NULL, use the default */
        public bool notNull;    /* True if there is a NOT NULL constraint */
        public bool isPrimKey;  /* True if this column is part of the PRIMARY KEY */
        public byte affinity;   /* One of the SQLITE_AFF_... values */
        public bool isHidden;   /* True if this column is 'hidden' */
    }
}
