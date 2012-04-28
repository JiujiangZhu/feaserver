using System.Diagnostics;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
#if !SQLITE_OMIT_VIEW
        // The parser calls this routine in order to create a new VIEW
        internal static void sqlite3CreateView(Parse pParse, Token pBegin, Token pName1, Token pName2, Select pSelect, int isTemp, int noErr)
        {
            var sFix = new DbFixer();
            var db = pParse.db;
            if (pParse.nVar > 0)
            {
                sqlite3ErrorMsg(pParse, "parameters are not allowed in views");
                sqlite3SelectDelete(db, ref pSelect);
                return;
            }
            Token pName = null;
            sqlite3StartTable(pParse, pName1, pName2, isTemp, 1, 0, noErr);
            var p = pParse.pNewTable;
            if (p == null || pParse.nErr != 0)
            {
                sqlite3SelectDelete(db, ref pSelect);
                return;
            }
            sqlite3TwoPartName(pParse, pName1, pName2, ref pName);
            var iDb = sqlite3SchemaToIndex(db, p.pSchema);
            if (sqlite3FixInit(sFix, pParse, iDb, "view", pName) != 0 && sqlite3FixSelect(sFix, pSelect) != 0)
            {
                sqlite3SelectDelete(db, ref pSelect);
                return;
            }
            // Make a copy of the entire SELECT statement that defines the view. This will force all the Expr.token.z values to be dynamically
            // allocated rather than point to the input string - which means that they will persist after the current sqlite3_exec() call returns.
            p.pSelect = sqlite3SelectDup(db, pSelect, EXPRDUP_REDUCE);
            sqlite3SelectDelete(db, ref pSelect);
            if (0 == db.init.busy)
                sqlite3ViewGetColumnNames(pParse, p);
            // Locate the end of the CREATE VIEW statement.  Make sEnd point to the end.
            var sEnd = pParse.sLastToken;
            if (Check.ALWAYS(sEnd.z[0] != 0) && sEnd.z[0] != ';')
                sEnd.z = sEnd.z.Substring(sEnd.n);
            sEnd.n = 0;
            var n = (int)(pBegin.z.Length - sEnd.z.Length);
            var z = pBegin.z;
            while (Check.ALWAYS(n > 0) && sqlite3Isspace(z[n - 1]))
                n--;
            sEnd.z = z.Substring(n - 1);
            sEnd.n = 1;
            // Use sqlite3EndTable() to add the view to the SQLITE_MASTER table
            sqlite3EndTable(pParse, null, sEnd, null);
            return;
        }
#else
        internal static void sqlite3CreateView(Parse pParse, Token pBegin, Token pName1, Token pName2, Select pSelect, int isTemp, int noErr) { }
#endif

#if !SQLITE_OMIT_VIEW || !SQLITE_OMIT_VIRTUALTABLE
        // The Table structure pTable is really a VIEW.  Fill in the names of the columns of the view in the pTable structure.  Return the number
        // of errors.  If an error is seen leave an error message in pParse.zErrMsg.
        internal static int sqlite3ViewGetColumnNames(Parse pParse, Table pTable)
        {
            int nErr = 0;
            var db = pParse.db;
            Debug.Assert(pTable != null);
#if !SQLITE_OMIT_VIRTUALTABLE
            if (sqlite3VtabCallConnect(pParse, pTable) != 0)
                return SQLITE_ERROR;
#endif
            if (IsVirtual(pTable))
                return 0;
#if !SQLITE_OMIT_VIEW
            // A positive nCol means the columns names for this view are already known.
            if (pTable.nCol > 0)
                return 0;
            // A negative nCol is a special marker meaning that we are currently trying to compute the column names.  If we enter this routine with
            // a negative nCol, it means two or more views form a loop, like this:
            //     CREATE VIEW one AS SELECT * FROM two;
            //     CREATE VIEW two AS SELECT * FROM one;
            // Actually, the error above is now caught prior to reaching this point. But the following test is still important as it does come up
            // in the following:
            //     CREATE TABLE main.ex1(a);
            //     CREATE TEMP VIEW ex1 AS SELECT a FROM ex1;
            //     SELECT * FROM temp.ex1;
            if (pTable.nCol < 0)
            {
                sqlite3ErrorMsg(pParse, "view %s is circularly defined", pTable.zName);
                return 1;
            }
            Debug.Assert(pTable.nCol >= 0);
            // If we get this far, it means we need to compute the table names. Note that the call to sqlite3ResultSetOfSelect() will expand any
            // "*" elements in the results set of the view and will assign cursors to the elements of the FROM clause.  But we do not want these changes
            // to be permanent.  So the computation is done on a copy of the SELECT statement that defines the view.
            Debug.Assert(pTable.pSelect != null);
            var pSel = sqlite3SelectDup(db, pTable.pSelect, 0);
            if (pSel != null)
            {
                var enableLookaside = db.lookaside.bEnabled;
                var n = pParse.nTab;
                sqlite3SrcListAssignCursors(pParse, pSel.pSrc);
                pTable.nCol = -1;
                db.lookaside.bEnabled = 0;
                var xAuth = db.xAuth;
                db.xAuth = 0;
                var pSelTab = sqlite3ResultSetOfSelect(pParse, pSel);
                db.xAuth = xAuth;
                db.lookaside.bEnabled = enableLookaside;
                pParse.nTab = n;
                if (pSelTab != null)
                {
                    Debug.Assert(pTable.aCol == null);
                    pTable.nCol = pSelTab.nCol;
                    pTable.aCol = pSelTab.aCol;
                    pSelTab.nCol = 0;
                    pSelTab.aCol = null;
                    sqlite3DeleteTable(db, ref pSelTab);
                    Debug.Assert(sqlite3SchemaMutexHeld(db, 0, pTable.pSchema));
                    pTable.pSchema.flags |= DB_UnresetViews;
                }
                else
                {
                    pTable.nCol = 0;
                    nErr++;
                }
                sqlite3SelectDelete(db, ref pSel);
            }
            else
                nErr++;
#endif
            return nErr;
        }
#endif

#if !SQLITE_OMIT_VIEW
        // Clear the column names from every VIEW in database idx.
        internal static void sqliteViewResetAll(sqlite3 db, int idx)
        {
            Debug.Assert(sqlite3SchemaMutexHeld(db, idx, null));
            if (!DbHasProperty(db, idx, DB_UnresetViews))
                return;
            for (var i = db.aDb[idx].pSchema.tblHash.first; i != null; i = i.next)
            {
                var pTab = (Table)i.data;// sqliteHashData( i );
                if (pTab.pSelect != null)
                {
                    sqliteDeleteColumnNames(db, pTab);
                    pTab.aCol = null;
                    pTab.nCol = 0;

                }
            }
            DbClearProperty(db, idx, DB_UnresetViews);
        }
#else
        static void sqliteViewResetAll(sqlite3 A, int B) { }
#endif
    }
}