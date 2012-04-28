using FeaServer.Engine.Core;
namespace FeaServer.Engine.Query
{
    public class VdbeCursor
    {
        public BtCursor pCursor;        // The cursor structure of the backend
        public Btree pBt;               // Separate file holding temporary table
        public KeyInfo pKeyInfo;        // Info about index keys needed by index cursors
        public int iDb;                 // Index of cursor database in db->aDb[] (or -1)
        public int pseudoTableReg;      // Register holding pseudotable content.
        public int nField;              // Number of fields in the header
        public bool zeroed;             // True if zeroed out and ready for reuse
        public bool rowidIsValid;       // True if lastRowid is valid
        public bool atFirst;            // True if pointing to first entry
        public bool useRandomRowid;     // Generate new record numbers semi-randomly
        public bool nullRow;            // True if pointing to a row with no data
        public bool deferredMoveto;     // A call to sqlite3BtreeMoveto() is needed
        public bool isTable;            // True if a table requiring integer keys
        public bool isIndex;            // True if an index containing keys only - no data
        public bool isOrdered;          // True if the underlying table is BTREE_UNORDERED
//#if !SQLITE_OMIT_VIRTUALTABLE
//        public sqlite3_vtab_cursor pVtabCursor; // The cursor for a virtual table
//        public sqlite3_module pModule;  // Module for cursor pVtabCursor
//#endif
        public long seqCount;           // Sequence counter
        public long movetoTarget;       // Argument to the deferred sqlite3BtreeMoveto()
        public long lastRowid;          // Last rowid from a Next or NextIdx operation

        // Result of last sqlite3BtreeMoveto() done by an OP_NotExists or OP_IsUnique opcode on this cursor.
        public int seekResult;

        // Cached information about the header for the data record that the cursor is currently pointing to.  Only valid if cacheStatus matches
        // Vdbe.cacheCtr.  Vdbe.cacheCtr will never take on the value of CACHE_STALE and so setting cacheStatus=CACHE_STALE guarantees that
        // the cache is out of date.
        // aRow might point to (ephemeral) data for the current row, or it might be NULL.
        public uint cacheStatus;        // Cache is valid if this matches Vdbe.cacheCtr
        public uint payloadSize;        // Total number of bytes in the record
        public uint[] aType;            // Type values for all entries in the record
        public uint[] aOffset;          // Cached offsets to the start of each columns data
        public int aRow;                // Pointer to Data for the current row, if all on one page

        public VdbeCursor Copy()
        {
            return (VdbeCursor)MemberwiseClone();
        }
    }
}