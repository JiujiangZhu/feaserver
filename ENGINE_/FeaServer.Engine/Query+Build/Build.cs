using System.Diagnostics;
using System;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // This routine is called when a new SQL statement is beginning to be parsed.  Initialize the pParse structure as needed.
        internal static void sqlite3BeginParse(Parse pParse, int explainFlag)
        {
            pParse.explain = (byte)explainFlag;
            pParse.nVar = 0;
        }

        // This routine is called after a single SQL statement has been parsed and a VDBE program to execute that statement has been
        // prepared.  This routine puts the finishing touches on the VDBE program and resets the pParse structure for the next parse.
        //
        // Note that if an error occurred, it might be the case that no VDBE code was generated.
        static void sqlite3FinishCoding(Parse pParse)
        {
            var db = pParse.db;
            if (pParse.nested != 0)
                return;
            if (pParse.nErr != 0)
                return;
            // Begin by generating some termination code at the end of the vdbe program
            var v = sqlite3GetVdbe(pParse);
            Debug.Assert(0 == pParse.isMultiWrite
#if DEBUG
 || sqlite3VdbeAssertMayAbort(v, pParse.mayAbort) != 0
#endif
);
            if (v != null)
            {
                sqlite3VdbeAddOp0(v, OP_Halt);
                // The cookie mask contains one bit for each database file open. (Bit 0 is for main, bit 1 is for temp, and so forth.)  Bits are
                // set for each database that is used.  Generate code to start a transaction on each used database and to verify the schema cookie
                // on each used database.
                if (pParse.cookieGoto > 0)
                {
                    sqlite3VdbeJumpHere(v, pParse.cookieGoto - 1);
                    for (uint iDb = 0, mask = 1; iDb < db.nDb; mask <<= 1, iDb++)
                    {
                        if ((mask & pParse.cookieMask) == 0)
                            continue;
                        sqlite3VdbeUsesBtree(v, iDb);
                        sqlite3VdbeAddOp2(v, OP_Transaction, iDb, (mask & pParse.writeMask) != 0);
                        if (db.init.busy == 0)
                        {
                            Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
                            sqlite3VdbeAddOp3(v, OP_VerifyCookie, iDb, pParse.cookieValue[iDb], (int)db.aDb[iDb].pSchema.iGeneration);
                        }
                    }
#if !SQLITE_OMIT_VIRTUALTABLE
                    for (var i = 0; i < pParse.nVtabLock; i++)
                    {
                        var vtab = sqlite3GetVTable(db, pParse.apVtabLock[i]);
                        sqlite3VdbeAddOp4(v, OP_VBegin, 0, 0, 0, vtab, P4_VTAB);
                    }
                    pParse.nVtabLock = 0;
#endif
                    // Once all the cookies have been verified and transactions opened, obtain the required table-locks. This is a no-op unless the
                    // shared-cache feature is enabled.
                    codeTableLocks(pParse);
                    // Initialize any AUTOINCREMENT data structures required.
                    sqlite3AutoincrementBegin(pParse);
                    // Finally, jump back to the beginning of the executable code.
                    sqlite3VdbeAddOp2(v, OP_Goto, 0, pParse.cookieGoto);
                }
            }

            // Get the VDBE program ready for execution
            if (v != null && Check.ALWAYS(pParse.nErr == 0))
            {
#if DEBUG
                var trace = ((db.flags & SQLITE_VdbeTrace) != 0 ? Console.Out : null);
                sqlite3VdbeTrace(v, trace);
#endif
                Debug.Assert(pParse.iCacheLevel == 0);  /* Disables and re-enables match */
                // A minimum of one cursor is required if autoincrement is used
                if (pParse.pAinc != null && pParse.nTab == 0)
                    pParse.nTab = 1;
                sqlite3VdbeMakeReady(v, pParse);
                pParse.rc = SQLITE_DONE;
                pParse.colNamesSet = 0;
            }
            else
                pParse.rc = SQLITE_ERROR;
            pParse.nTab = 0;
            pParse.nMem = 0;
            pParse.nSet = 0;
            pParse.nVar = 0;
            pParse.cookieMask = 0;
            pParse.cookieGoto = 0;
        }

        private static readonly Object _nestingLock = new Object();

        // Run the parser and code generator recursively in order to generate code for the SQL statement given onto the end of the pParse context
        // currently under construction.  When the parser is run recursively this way, the final OP_Halt is not appended and other initialization
        // and finalization steps are omitted because those are handling by the outermost parser.
        //
        // Not everything is nestable.  This facility is designed to permit INSERT, UPDATE, and DELETE operations against SQLITE_MASTER.  Use
        // care if you decide to try to use this routine for some other purposes.
        internal static void sqlite3NestedParse(Parse pParse, string zFormat, params object[] ap)
        {
            var db = pParse.db;
            if (pParse.nErr != 0)
                return;
            Debug.Assert(pParse.nested < 10);  // Nesting should only be of limited depth
            lock (lock_va_list)
            {
                va_start(ap, zFormat);
                var zSql = sqlite3VMPrintf(db, zFormat, ap);
                va_end(ref ap);
            }
            lock (_nestingLock)
            {
                pParse.nested++;
                pParse.SaveMembers();
                pParse.ResetMembers();
                string zErrMsg = string.Empty;
                sqlite3RunParser(pParse, zSql, ref zErrMsg);
                sqlite3DbFree(db, ref zErrMsg);
                sqlite3DbFree(db, ref zSql);
                pParse.RestoreMembers();
                pParse.nested--;
            }
        }

        // This routine is called when a commit occurs.
        internal static void sqlite3CommitInternalChanges(sqlite3 db)
        {
            db.flags &= ~SQLITE_InternChanges;
        }

        // Open the sqlite_master table stored in database number iDb for writing. The table is opened using cursor 0.
        internal static void sqlite3OpenMasterTable(Parse p, int iDb)
        {
            var v = sqlite3GetVdbe(p);
            sqlite3TableLock(p, iDb, MASTER_ROOT, 1, SCHEMA_TABLE(iDb));
            sqlite3VdbeAddOp3(v, OP_OpenWrite, 0, MASTER_ROOT, iDb);
            sqlite3VdbeChangeP4(v, -1, (int)5, P4_INT32);  /* 5 column table */
            if (p.nTab == 0)
                p.nTab = 1;
        }

        // Make sure the TEMP database is open and available for use.  Return the number of errors.  Leave any error messages in the pParse structure.
        internal static int sqlite3OpenTempDatabase(Parse pParse)
        {
            var db = pParse.db;
            if (db.aDb[1].pBt == null && pParse.explain == 0)
            {
                Btree pBt = null;
                var rc = sqlite3BtreeOpen(db.pVfs, null, db, ref pBt, 0, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_EXCLUSIVE | SQLITE_OPEN_DELETEONCLOSE | SQLITE_OPEN_TEMP_DB);
                if (rc != SQLITE_OK)
                {
                    sqlite3ErrorMsg(pParse, "unable to open a temporary database file for storing temporary tables");
                    pParse.rc = rc;
                    return 1;
                }
                db.aDb[1].pBt = pBt;
                Debug.Assert(db.aDb[1].pSchema != null);
                sqlite3BtreeSetPageSize(pBt, db.nextPagesize, -1, 0);
            }
            return 0;
        }

        // Code an OP_Halt that causes the vdbe to return an SQLITE_CONSTRAINT error. The onError parameter determines which (if any) of the statement
        // and/or current transaction is rolled back.
        internal static void sqlite3HaltConstraint(Parse pParse, int onError, string p4, int p4type)
        {
            var v = sqlite3GetVdbe(pParse);
            if (onError == OE_Abort)
                sqlite3MayAbort(pParse);
            sqlite3VdbeAddOp4(v, OP_Halt, SQLITE_CONSTRAINT, onError, 0, p4, p4type);
        }
        internal static void sqlite3HaltConstraint(Parse pParse, int onError, byte[] p4, int p4type)
        {
            var v = sqlite3GetVdbe(pParse);
            if (onError == OE_Abort)
                sqlite3MayAbort(pParse);
            sqlite3VdbeAddOp4(v, OP_Halt, SQLITE_CONSTRAINT, onError, 0, p4, p4type);
        }

        // Return a dynamicly allocated KeyInfo structure that can be used with OP_OpenRead or OP_OpenWrite to access database index pIdx.
        //
        // If successful, a pointer to the new structure is returned. In this case the caller is responsible for calling sqlite3DbFree(db, ) on the returned
        // pointer. If an error occurs (out of memory or missing collation sequence), NULL is returned and the state of pParse updated to reflect
        // the error.
        internal static KeyInfo sqlite3IndexKeyinfo(Parse pParse, Index pIdx)
        {
            var nCol = pIdx.nColumn;
            var db = pParse.db;
            var pKey = new KeyInfo();
            if (pKey != null)
            {
                pKey.db = pParse.db;
                pKey.aSortOrder = new byte[nCol];
                pKey.aColl = new CollSeq[nCol];
                for (var i = 0; i < nCol; i++)
                {
                    var zColl = pIdx.azColl[i];
                    Debug.Assert(zColl != null);
                    pKey.aColl[i] = sqlite3LocateCollSeq(pParse, zColl);
                    pKey.aSortOrder[i] = pIdx.aSortOrder[i];
                }
                pKey.nField = (ushort)nCol;
            }
            if (pParse.nErr != 0)
            {
                pKey = null;
                sqlite3DbFree(db, ref pKey);
            }
            return pKey;
        }
    }
}