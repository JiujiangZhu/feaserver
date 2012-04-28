using System.Diagnostics;
using System;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // Locate the in-memory structure that describes a particular database table given the name of that table and (optionally) the name of the
        // database containing the table.  Return NULL if not found.
        //
        // If zDatabase is 0, all databases are searched for the table and the first matching table is returned.  (No checking for duplicate table
        // names is done.)  The search order is TEMP first, then MAIN, then any auxiliary databases added using the ATTACH command.
        //
        // See also sqlite3LocateTable().
        internal static Table sqlite3FindTable(sqlite3 db, string zName, string zDatabase)
        {
            Table p = null;
            Debug.Assert(zName != null);
            var nName = sqlite3Strlen30(zName);
            // All mutexes are required for schema access.  Make sure we hold them.
            Debug.Assert(zDatabase != null || sqlite3BtreeHoldsAllMutexes(db));
            for (var i = OMIT_TEMPDB; i < db.nDb; i++)
            {
                var j = (i < 2 ? i ^ 1 : i);   // Search TEMP before MAIN
                if (zDatabase != null && !zDatabase.Equals(db.aDb[j].zName, StringComparison.InvariantCultureIgnoreCase))
                    continue;
                Debug.Assert(sqlite3SchemaMutexHeld(db, j, null));
                p = sqlite3HashFind(db.aDb[j].pSchema.tblHash, zName, nName, (Table)null);
                if (p != null)
                    break;
            }
            return p;
        }

        // Locate the in-memory structure that describes a particular database table given the name of that table and (optionally) the name of the
        // database containing the table.  Return NULL if not found.  Also leave an error message in pParse.zErrMsg.
        //
        // The difference between this routine and sqlite3FindTable() is that this routine leaves an error message in pParse.zErrMsg where
        // sqlite3FindTable() does not.
        internal static Table sqlite3LocateTable(Parse pParse, int isView, string zName, string zDbase)
        {
            // Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
            if (SQLITE_OK != sqlite3ReadSchema(pParse))
                return null;
            var p = sqlite3FindTable(pParse.db, zName, zDbase);
            if (p == null)
            {
                var zMsg = (isView != 0 ? "no such view" : "no such table");
                if (zDbase != null)
                    sqlite3ErrorMsg(pParse, "%s: %s.%s", zMsg, zDbase, zName);
                else
                    sqlite3ErrorMsg(pParse, "%s: %s", zMsg, zName);
                pParse.checkSchema = 1;
            }
            return p;
        }

        // Locate the in-memory structure that describes a particular index given the name of that index
        // and the name of the database that contains the index. Return NULL if not found.
        //
        // If zDatabase is 0, all databases are searched for the table and the first matching index is returned.  (No checking
        // for duplicate index names is done.)  The search order is TEMP first, then MAIN, then any auxiliary databases added
        // using the ATTACH command.
        internal static Index sqlite3FindIndex(sqlite3 db, string zName, string zDb)
        {
            Index p = null;
            int nName = sqlite3Strlen30(zName);
            // All mutexes are required for schema access.  Make sure we hold them.
            Debug.Assert(zDb != null || sqlite3BtreeHoldsAllMutexes(db));
            for (var i = OMIT_TEMPDB; i < db.nDb; i++)
            {
                var j = (i < 2 ? i ^ 1 : i);  // Search TEMP before MAIN
                var pSchema = db.aDb[j].pSchema;
                Debug.Assert(pSchema != null);
                if (zDb != null && !zDb.Equals(db.aDb[j].zName, StringComparison.InvariantCultureIgnoreCase))
                    continue;
                Debug.Assert(sqlite3SchemaMutexHeld(db, j, null));
                p = sqlite3HashFind(pSchema.idxHash, zName, nName, (Index)null);
                if (p != null)
                    break;
            }
            return p;
        }

        // Given a token, return a string that consists of the text of that token.  Space to hold the returned string
        // is obtained from sqliteMalloc() and must be freed by the calling function.
        //
        // Any quotation marks (ex:  "name", 'name', [name], or `name`) that surround the body of the token are removed.
        //
        // Tokens are often just pointers into the original SQL text and so are not \000 terminated and are not persistent.  The returned string
        // is \000 terminated and is persistent.
        internal static string sqlite3NameFromToken(sqlite3 db, Token pName)
        {
            string zName;
            if (pName != null && pName.z != null)
            {
                zName = pName.z.Substring(0, pName.n);
                sqlite3Dequote(ref zName);
            }
            else
                return null;
            return zName;
        }

        // Parameter zName points to a nul-terminated buffer containing the name of a database ("main", "temp" or the name of an attached db). This
        // function returns the index of the named database in db->aDb[], or -1 if the named db cannot be found.
        internal static int sqlite3FindDbName(sqlite3 db, string zName)
        {
            var i = -1;    /* Database number */
            if (zName != null)
            {
                int n = sqlite3Strlen30(zName);
                for (i = (db.nDb - 1); i >= 0; i--)
                {
                    var pDb = db.aDb[i];
                    if ((OMIT_TEMPDB == 0 || i != 1) && n == sqlite3Strlen30(pDb.zName) && pDb.zName.Equals(zName, StringComparison.InvariantCultureIgnoreCase))
                        break;
                }
            }
            return i;
        }

        // The token *pName contains the name of a database (either "main" or "temp" or the name of an attached db). This routine returns the
        // index of the named database in db->aDb[], or -1 if the named db does not exist.
        internal static int sqlite3FindDb(sqlite3 db, Token pName)
        {
            var zName = sqlite3NameFromToken(db, pName);
            var i = sqlite3FindDbName(db, zName);
            sqlite3DbFree(db, ref zName);
            return i;
        }

        // The table or view or trigger name is passed to this routine via tokens pName1 and pName2. If the table name was fully qualified, for example:
        // CREATE TABLE xxx.yyy (...);
        // Then pName1 is set to "xxx" and pName2 "yyy". On the other hand if the table name is not fully qualified, i.e.:
        // CREATE TABLE yyy(...);
        // Then pName1 is set to "yyy" and pName2 is "".
        // This routine sets the ppUnqual pointer to point at the token (pName1 or pName2) that stores the unqualified table name.  The index of the
        // database "xxx" is returned.
        internal static int sqlite3TwoPartName(Parse pParse, Token pName1, Token pName2, ref Token pUnqual)
        {
            sqlite3 db = pParse.db;
            int iDb;
            if (Check.ALWAYS(pName2 != null) && pName2.n > 0)
            {
                if (db.init.busy != 0)
                {
                    sqlite3ErrorMsg(pParse, "corrupt database");
                    pParse.nErr++;
                    return -1;
                }
                pUnqual = pName2;
                iDb = sqlite3FindDb(db, pName1);
                if (iDb < 0)
                {
                    sqlite3ErrorMsg(pParse, "unknown database %T", pName1);
                    pParse.nErr++;
                    return -1;
                }
            }
            else
            {
                Debug.Assert(db.init.iDb == 0 || db.init.busy != 0);
                iDb = db.init.iDb;
                pUnqual = pName1;
            }
            return iDb;
        }

        // This routine is used to check if the UTF-8 string zName is a legal unqualified name for a new schema object (vtable, index, view or
        // trigger). All names are legal except those that begin with the string "sqlite_" (in upper, lower or mixed case). This portion of the namespace
        // is reserved for internal use.
        internal static int sqlite3CheckObjectName(Parse pParse, string zName)
        {
            if (0 == pParse.db.init.busy && pParse.nested == 0 && (pParse.db.flags & SQLITE_WriteSchema) == 0 && zName.StartsWith("sqlite_", System.StringComparison.InvariantCultureIgnoreCase))
            {
                sqlite3ErrorMsg(pParse, "object name reserved for internal use: %s", zName);
                return SQLITE_ERROR;
            }
            return SQLITE_OK;
        }

        // This function returns the collation sequence for database native text encoding identified by the string zName, length nName.
        //
        // If the requested collation sequence is not available, or not available in the database native encoding, the collation factory is invoked to
        // request it. If the collation factory does not supply such a sequence, and the sequence is available in another text encoding, then that is
        // returned instead.
        //
        // If no versions of the requested collations sequence are available, or another error occurs, NULL is returned and an error message written into
        // pParse.
        //
        // This routine is a wrapper around sqlite3FindCollSeq().  This routine invokes the collation factory if the named collation cannot be found
        // and generates an error message.
        //
        // See also: sqlite3FindCollSeq(), sqlite3GetCollSeq()
        internal static CollSeq sqlite3LocateCollSeq(Parse pParse, string zName)
        {
            var db = pParse.db;
            var enc = db.aDb[0].pSchema.enc;// ENC(db);
            var initbusy = db.init.busy;
            var pColl = sqlite3FindCollSeq(db, enc, zName, initbusy);
            if (0 == initbusy && (pColl == null || pColl.xCmp == null))
            {
                pColl = sqlite3GetCollSeq(db, enc, pColl, zName);
                if (pColl == null)
                    sqlite3ErrorMsg(pParse, "no such collation sequence: %s", zName);
            }
            return pColl;
        }
    }
}