using System.Diagnostics;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // Begin a transaction
        internal static void sqlite3BeginTransaction(Parse pParse, Parser.TK type)
        {
            Debug.Assert(pParse != null);
            var db = pParse.db;
            Debug.Assert(db != null);
            if (sqlite3AuthCheck(pParse, SQLITE_TRANSACTION, "BEGIN", null, null) != 0)
                return;
            var v = sqlite3GetVdbe(pParse);
            if (v == null)
                return;
            if (type != Parser.TK.DEFERRED)
                for (var i = 0; i < db.nDb; i++)
                {
                    sqlite3VdbeAddOp2(v, OP_Transaction, i, type == Parser.TK.EXCLUSIVE ? 2 : 1);
                    sqlite3VdbeUsesBtree(v, i);
                }
            sqlite3VdbeAddOp2(v, OP_AutoCommit, 0, 0);
        }

        // Commit a transaction
        internal static void sqlite3CommitTransaction(Parse pParse)
        {
            Debug.Assert(pParse != null);
            var db = pParse.db;
            Debug.Assert(db != null);
            if (sqlite3AuthCheck(pParse, SQLITE_TRANSACTION, "COMMIT", null, null) != 0)
                return;
            var v = sqlite3GetVdbe(pParse);
            if (v != null)
                sqlite3VdbeAddOp2(v, OP_AutoCommit, 1, 0);
        }

        // Rollback a transaction
        internal static void sqlite3RollbackTransaction(Parse pParse)
        {
            Debug.Assert(pParse != null);
            var db = pParse.db;
            Debug.Assert(db != null);
            if (sqlite3AuthCheck(pParse, SQLITE_TRANSACTION, "ROLLBACK", null, null) != 0)
                return;
            var v = sqlite3GetVdbe(pParse);
            if (v != null)
                sqlite3VdbeAddOp2(v, OP_AutoCommit, 1, 1);
        }

        // This function is called by the parser when it parses a command to create, release or rollback an SQL savepoint.
        const string[] az = { "BEGIN", "RELEASE", "ROLLBACK" };
        internal static void sqlite3Savepoint(Parse pParse, int op, Token pName)
        {
            var zName = sqlite3NameFromToken(pParse.db, pName);
            if (zName != null)
            {
                var v = sqlite3GetVdbe(pParse);
                Debug.Assert(!SAVEPOINT_BEGIN && SAVEPOINT_RELEASE == 1 && SAVEPOINT_ROLLBACK == 2);
                if (null == v || sqlite3AuthCheck(pParse, SQLITE_SAVEPOINT, az[op], zName, 0))
                {
                    sqlite3DbFree(pParse.db, ref zName);
                    return;
                }
                sqlite3VdbeAddOp4(v, OP_Savepoint, op, 0, 0, zName, P4_DYNAMIC);
            }
        }
    }
}