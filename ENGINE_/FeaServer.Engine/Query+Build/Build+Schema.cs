using System;
using System.Diagnostics;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // Erase all schema information from the in-memory hash tables of a single database.  This routine is called to reclaim memory
        // before the database closes.  It is also called during a rollback if there were schema changes during the transaction or if a
        // schema-cookie mismatch occurs.
        //
        // If iDb<0 then reset the internal schema tables for all database files.  If iDb>=0 then reset the internal schema for only the
        // single file indicated.
        internal static void sqlite3ResetInternalSchema(sqlite3 db, int iDb)
        {
            Debug.Assert(iDb < db.nDb);
            if (iDb >= 0)
            {
                // Case 1:  Reset the single schema identified by iDb
                var pDb = db.aDb[iDb];
                Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
                Debug.Assert(pDb.pSchema != null);
                sqlite3SchemaClear(pDb.pSchema);
                // If any database other than TEMP is reset, then also reset TEMP since TEMP might be holding triggers that reference tables in the other database.
                if (iDb != 1)
                {
                    pDb = db.aDb[1];
                    Debug.Assert(pDb.pSchema != null);
                    sqlite3SchemaClear(pDb.pSchema);
                }
                return;
            }
            // Case 2 (from here to the end): Reset all schemas for all attached databases.
            Debug.Assert(iDb < 0);
            sqlite3BtreeEnterAll(db);
            for (var i = 0; i < db.nDb; i++)
            {
                var pDb = db.aDb[i];
                if (pDb.pSchema != null)
                    sqlite3SchemaClear(pDb.pSchema);
            }
            db.flags &= ~SQLITE_InternChanges;
            sqlite3VtabUnlockList(db);
            sqlite3BtreeLeaveAll(db);
            // If one or more of the auxiliary database files has been closed, then remove them from the auxiliary database list.  We take the
            // opportunity to do this here since we have just deleted all of the schema hash tables and therefore do not have to make any changes
            // to any of those tables.
            int j;
            for (var i = j = 2; i < db.nDb; i++)
            {
                var pDb = db.aDb[i];
                if (pDb.pBt == null)
                {
                    sqlite3DbFree(db, ref pDb.zName);
                    continue;
                }
                if (j < i)
                    db.aDb[j] = db.aDb[i];
                j++;
            }
            if (db.nDb != j)
                db.aDb[j] = new Db();
            db.nDb = j;
            if (db.nDb <= 2 && db.aDb != db.aDbStatic)
                Array.Copy(db.aDb, db.aDbStatic, 2);
        }

        // Generate code that will increment the schema cookie.
        //
        // The schema cookie is used to determine when the schema for the database changes.  After each schema change, the cookie value
        // changes.  When a process first reads the schema it records the cookie.  Thereafter, whenever it goes to access the database,
        // it checks the cookie to make sure the schema has not changed since it was last read.
        //
        // This plan is not completely bullet-proof.  It is possible for the schema to change multiple times and for the cookie to be
        // set back to prior value.  But schema changes are infrequent and the probability of hitting the same cookie value is only
        // 1 chance in 2^32.  So we're safe enough.
        internal static void sqlite3ChangeCookie(Parse pParse, int iDb)
        {
            var r1 = sqlite3GetTempReg(pParse);
            var db = pParse.db;
            var v = pParse.pVdbe;
            Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
            sqlite3VdbeAddOp2(v, OP_Integer, db.aDb[iDb].pSchema.schema_cookie + 1, r1);
            sqlite3VdbeAddOp3(v, OP_SetCookie, iDb, BTREE_SCHEMA_VERSION, r1);
            sqlite3ReleaseTempReg(pParse, r1);
        }

        // This function is called by the VDBE to adjust the internal schema used by SQLite when the btree layer moves a table root page. The
        // root-page of a table or index in database iDb has changed from iFrom to iTo.
#if !SQLITE_OMIT_AUTOVACUUM
        internal static void sqlite3RootPageMoved(sqlite3 db, int iDb, int iFrom, int iTo)
        {
            Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
            var pDb = db.aDb[iDb];
            var pHash = pDb.pSchema.tblHash;
            for (var pElem = pHash.first; pElem != null; pElem = pElem.next)
            {
                var pTab = (Table)pElem.data;
                if (pTab.tnum == iFrom)
                    pTab.tnum = iTo;
            }
            pHash = pDb.pSchema.idxHash;
            for (var pElem = pHash.first; pElem != null; pElem = pElem.next)
            {
                var pIdx = (Index)pElem.data;
                if (pIdx.tnum == iFrom)
                    pIdx.tnum = iTo;
            }
        }
#endif


    // Generate VDBE code that will verify the schema cookie and start a read-transaction for all named database files.
    //
    // It is important that all schema cookies be verified and all read transactions be started before anything else happens in
    // the VDBE program.  But this routine can be called after much other code has been generated.  So here is what we do:
    //
    // The first time this routine is called, we code an OP_Goto that will jump to a subroutine at the end of the program.  Then we
    // record every database that needs its schema verified in the pParse.cookieMask field.  Later, after all other code has been
    // generated, the subroutine that does the cookie verifications and starts the transactions will be coded and the OP_Goto P2 value
    // will be made to point to that subroutine.  The generation of the cookie verification subroutine code happens in sqlite3FinishCoding().
    //
    // If iDb<0 then code the OP_Goto only - don't set flag to verify the schema on any databases.  This can be used to position the OP_Goto
    // early in the code, before we know if any database tables will be used.
        internal static void sqlite3CodeVerifySchema(Parse pParse, int iDb)
        {
            var pToplevel = sqlite3ParseToplevel(pParse);
            if (pToplevel.cookieGoto == 0)
            {
                var v = sqlite3GetVdbe(pToplevel);
                if (v == null)
                    return;  // This only happens if there was a prior error
                pToplevel.cookieGoto = sqlite3VdbeAddOp2(v, OP_Goto, 0, 0) + 1;
            }
            if (iDb >= 0)
            {
                var db = pToplevel.db;
                Debug.Assert(iDb < db.nDb);
                Debug.Assert(db.aDb[iDb].pBt != null || iDb == 1);
                Debug.Assert(iDb < SQLITE_MAX_ATTACHED + 2);
                Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
                yDbMask mask = ((yDbMask)1) << iDb;
                if ((pToplevel.cookieMask & mask) == 0)
                {
                    pToplevel.cookieMask |= mask;
                    pToplevel.cookieValue[iDb] = db.aDb[iDb].pSchema.schema_cookie;
                    if (0 == OMIT_TEMPDB && iDb == 1)
                        sqlite3OpenTempDatabase(pToplevel);
                }
            }
        }

        // If argument zDb is NULL, then call sqlite3CodeVerifySchema() for each  attached database. Otherwise, invoke it for the database named zDb only.
        internal static void sqlite3CodeVerifyNamedSchema(Parse pParse, string zDb)
        {
            var db = pParse.db;
            for (var i = 0; i < db.nDb; i++)
            {
                var pDb = db.aDb[i];
                if (pDb.pBt != null && (null == zDb || 0 == zDb.CompareTo(pDb.zName)))
                    sqlite3CodeVerifySchema(pParse, i);
            }
        }
    }
}