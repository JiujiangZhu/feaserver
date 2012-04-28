using System.Diagnostics;
using System;
namespace FeaServer.Engine.Query
{
#if !SQLITE_OMIT_REINDEX
    public partial class Build
    {
        // Check to see if pIndex uses the collating sequence pColl.  Return true if it does and false if it does not.
        internal static bool collationMatch(string zColl, Index pIndex)
        {
            Debug.Assert(zColl != null);
            for (var i = 0; i < pIndex.nColumn; i++)
            {
                var z = pIndex.azColl[i];
                Debug.Assert(z != null);
                if (z.Equals(zColl, StringComparison.InvariantCultureIgnoreCase))
                    return true;
            }
            return false;
        }

        // Recompute all indices of pTab that use the collating sequence pColl.
        // If pColl == null then recompute all indices of pTab.
        internal static void reindexTable(Parse pParse, Table pTab, string zColl)
        {
            for (var pIndex = pTab.pIndex; pIndex != null; pIndex = pIndex.pNext)
                if (zColl == null || collationMatch(zColl, pIndex))
                {
                    var iDb = sqlite3SchemaToIndex(pParse.db, pTab.pSchema);
                    sqlite3BeginWriteOperation(pParse, 0, iDb);
                    sqlite3RefillIndex(pParse, pIndex, -1);
                }
        }

        // Recompute all indices of all tables in all databases where the indices use the collating sequence pColl.  If pColl == null then recompute
        // all indices everywhere.
        internal static void reindexDatabases(Parse pParse, string zColl)
        {
            var db = pParse.db;
            Debug.Assert(sqlite3BtreeHoldsAllMutexes(db));  // Needed for schema access
            for (var iDb = 0; iDb < db.nDb; iDb++)
            {
                var pDb = db.aDb[iDb];
                Debug.Assert(pDb != null);
                for (var k = pDb.pSchema.tblHash.first; k != null; k = k.next)
                {
                    var pTab = (Table)k.data;
                    reindexTable(pParse, pTab, zColl);
                }
            }
        }

        // Generate code for the REINDEX command.
        //        REINDEX                            -- 1
        //        REINDEX  <collation>               -- 2
        //        REINDEX  ?<database>.?<tablename>  -- 3
        //        REINDEX  ?<database>.?<indexname>  -- 4
        // Form 1 causes all indices in all attached databases to be rebuilt.
        // Form 2 rebuilds all indices in all databases that use the named
        // collating function.  Forms 3 and 4 rebuild the named index or all indices associated with the named table.
        internal static void sqlite3Reindex(Parse pParse, int null_2, int null_3) { sqlite3Reindex(pParse, null, null); }
        internal static void sqlite3Reindex(Parse pParse, Token pName1, Token pName2)
        {
            var db = pParse.db;     
            var pObjName = new Token(); 
            // Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
            if (SQLITE_OK != sqlite3ReadSchema(pParse))
                return;
            if (pName1 == null)
            {
                reindexDatabases(pParse, null);
                return;
            }
            else if (Check.NEVER(pName2 == null) || pName2.z == null || pName2.z.Length == 0)
            {
                Debug.Assert(pName1.z != null);
                var zColl = sqlite3NameFromToken(pParse.db, pName1);
                if (zColl == null)
                    return;
                var pColl = sqlite3FindCollSeq(db, ENC(db), zColl, 0);
                if (pColl != null)
                {
                    reindexDatabases(pParse, zColl);
                    sqlite3DbFree(db, ref zColl);
                    return;
                }
                sqlite3DbFree(db, ref zColl);
            }
            var iDb = sqlite3TwoPartName(pParse, pName1, pName2, ref pObjName);
            if (iDb < 0)
                return;
            var z = sqlite3NameFromToken(db, pObjName);
            if (z == null)
                return;
            var zDb = db.aDb[iDb].zName;
            var pTab = sqlite3FindTable(db, z, zDb);
            if (pTab != null)
            {
                reindexTable(pParse, pTab, null);
                sqlite3DbFree(db, ref z);
                return;
            }
            var pIndex = sqlite3FindIndex(db, z, zDb);
            sqlite3DbFree(db, ref z);
            if (pIndex != null)
            {
                sqlite3BeginWriteOperation(pParse, 0, iDb);
                sqlite3RefillIndex(pParse, pIndex, -1);
                return;
            }
            sqlite3ErrorMsg(pParse, "unable to identify the object to be reindexed");
        }
#endif
    }
}