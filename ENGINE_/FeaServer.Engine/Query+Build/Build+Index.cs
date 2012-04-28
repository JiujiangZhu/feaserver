using System.Diagnostics;
using System.Text;
using System;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // This routine is called when an INITIALLY IMMEDIATE or INITIALLY DEFERRED clause is seen as part of a foreign key definition.  The isDeferred
        // parameter is 1 for INITIALLY DEFERRED and 0 for INITIALLY IMMEDIATE. The behavior of the most recently created foreign key is adjusted
        // accordingly.
        internal static void sqlite3DeferForeignKey(Parse pParse, int isDeferred)
        {
#if !SQLITE_OMIT_FOREIGN_KEY
            Table pTab;
            FKey pFKey;
            if ((pTab = pParse.pNewTable) == null || (pFKey = pTab.pFKey) == null)
                return;
            Debug.Assert(isDeferred == 0 || isDeferred == 1);
            pFKey.isDeferred = (byte)isDeferred;
#endif
        }

        // Generate code that will erase and refill index pIdx.  This is used to initialize a newly created index or to recompute the
        // content of an index in response to a REINDEX command.
        //
        // if memRootPage is not negative, it means that the index is newly created.  The register specified by memRootPage contains the
        // root page number of the index.  If memRootPage is negative, then the index already exists and must be cleared before being refilled and
        // the root page number of the index is taken from pIndex.tnum.
        internal static void sqlite3RefillIndex(Parse pParse, Index pIndex, int memRootPage)
        {
            var pTab = pIndex.pTable;
            var iTab = pParse.nTab++;
            var iIdx = pParse.nTab++;
            var db = pParse.db;
            var iDb = sqlite3SchemaToIndex(db, pIndex.pSchema);
            if (sqlite3AuthCheck(pParse, SQLITE_REINDEX, pIndex.zName, 0, db.aDb[iDb].zName))
                return;
            // Require a write-lock on the table to perform this operation
            sqlite3TableLock(pParse, iDb, pTab.tnum, 1, pTab.zName);
            var v = sqlite3GetVdbe(pParse);
            if (v == null)
                return;
            int tnum; // Root page of index
            if (memRootPage >= 0)
                tnum = memRootPage;
            else
            {
                tnum = pIndex.tnum;
                sqlite3VdbeAddOp2(v, OP_Clear, tnum, iDb);
            }
            var pKey = sqlite3IndexKeyinfo(pParse, pIndex);
            sqlite3VdbeAddOp4(v, OP_OpenWrite, iIdx, tnum, iDb, pKey, P4_KEYINFO_HANDOFF);
            if (memRootPage >= 0)
                sqlite3VdbeChangeP5(v, 1);
            sqlite3OpenTable(pParse, iTab, iDb, pTab, OP_OpenRead);
            var addr1 = sqlite3VdbeAddOp2(v, OP_Rewind, iTab, 0);
            var regRecord = sqlite3GetTempReg(pParse);
            var regIdxKey = sqlite3GenerateIndexKey(pParse, pIndex, iTab, regRecord, true);
            if (pIndex.onError != OE_None)
            {
                var regRowid = regIdxKey + pIndex.nColumn;
                var j2 = sqlite3VdbeCurrentAddr(v) + 2;
                var pRegKey = regIdxKey;
                // The registers accessed by the OP_IsUnique opcode were allocated using sqlite3GetTempRange() inside of the sqlite3GenerateIndexKey()
                // call above. Just before that function was freed they were released (made available to the compiler for reuse) using
                // sqlite3ReleaseTempRange(). So in some ways having the OP_IsUnique opcode use the values stored within seems dangerous. However, since
                // we can be sure that no other temp registers have been allocated since sqlite3ReleaseTempRange() was called, it is safe to do so.
                sqlite3VdbeAddOp4(v, OP_IsUnique, iIdx, j2, regRowid, pRegKey, P4_INT32);
                sqlite3HaltConstraint(pParse, OE_Abort, "indexed columns are not unique", P4_STATIC);
            }
            sqlite3VdbeAddOp2(v, OP_IdxInsert, iIdx, regRecord);
            sqlite3VdbeChangeP5(v, OPFLAG_USESEEKRESULT);
            sqlite3ReleaseTempReg(pParse, regRecord);
            sqlite3VdbeAddOp2(v, OP_Next, iTab, addr1 + 1);
            sqlite3VdbeJumpHere(v, addr1);
            sqlite3VdbeAddOp1(v, OP_Close, iTab);
            sqlite3VdbeAddOp1(v, OP_Close, iIdx);
        }

        // Create a new index for an SQL table.  pName1.pName2 is the name of the index and pTblList is the name of the table that is to be indexed.  Both will
        // be NULL for a primary key or an index that is created to satisfy a UNIQUE constraint.  If pTable and pIndex are NULL, use pParse.pNewTable
        // as the table to be indexed.  pParse.pNewTable is a table that is currently being constructed by a CREATE TABLE statement.
        //
        // pList is a list of columns to be indexed.  pList will be NULL if this is a primary key or unique-constraint on the most recent column added
        // to the table currently under construction.
        //
        // If the index is created successfully, return a pointer to the new Index structure. This is used by sqlite3AddPrimaryKey() to mark the index
        // as the tables primary key (Index.autoIndex==2).
        static Index sqlite3CreateIndex(Parse pParse, int null_2, int null_3, int null_4, int null_5, int onError, int null_7, int null_8, int sortOrder, int ifNotExist) { return sqlite3CreateIndex(pParse, null, null, null, null, onError, null, null, sortOrder, ifNotExist); }
        static Index sqlite3CreateIndex(Parse pParse, int null_2, int null_3, int null_4, ExprList pList, int onError, int null_7, int null_8, int sortOrder, int ifNotExist) { return sqlite3CreateIndex(pParse, null, null, null, pList, onError, null, null, sortOrder, ifNotExist); }
        static Index sqlite3CreateIndex(Parse pParse, Token pName1, Token pName2, SrcList pTblName, ExprList pList, int onError, Token pStart, Token pEnd, int sortOrder, int ifNotExist)
        {
            Index pRet = null;            /* Pointer to return */
            Table pTab = null;            /* Table to be indexed */
            Index pIndex = null;          /* The index to be created */
            string zName = null;          /* Name of the index */
            int nName;                    /* Number of characters in zName */
            var nullId = new Token();
            var sFix = new DbFixer();
            int sortOrderMask;            /* 1 to honor DESC in index.  0 to ignore. */
            var db = pParse.db;
            Db pDb;                       /* The specific table containing the indexed database */
            Token pName = null;           /* Unqualified name of the index to create */
            ExprList_item pListItem;      /* For looping over pList */
            int nCol;
            int nExtra = 0;
            var zExtra = new StringBuilder();

            Debug.Assert(pStart == null || pEnd != null); /* pEnd must be non-NULL if pStart is */
            Debug.Assert(pParse.nErr == 0);      /* Never called with prior errors */
            if (IN_DECLARE_VTAB(pParse))
                goto exit_create_index;
            if (SQLITE_OK != sqlite3ReadSchema(pParse))
                goto exit_create_index;
            // Find the table that is to be indexed.  Return early if not found.
            if (pTblName != null)
            {
                // Use the two-part index name to determine the database to search for the table. 'Fix' the table name to this db
                // before looking up the table.
                Debug.Assert(pName1 != null && pName2 != null);
                var iDb = sqlite3TwoPartName(pParse, pName1, pName2, ref pName);
                if (iDb < 0)
                    goto exit_create_index;
#if !SQLITE_OMIT_TEMPDB
                // If the index name was unqualified, check if the the table is a temp table. If so, set the database to 1. Do not do this
                // if initialising a database schema.
                if (0 == db.init.busy)
                {
                    pTab = sqlite3SrcListLookup(pParse, pTblName);
                    if (pName2.n == 0 && pTab != null && pTab.pSchema == db.aDb[1].pSchema)
                        iDb = 1;
                }
#endif
                if (sqlite3FixInit(sFix, pParse, iDb, "index", pName) != 0 && sqlite3FixSrcList(sFix, pTblName) != 0)
                    // Because the parser constructs pTblName from a single identifier, sqlite3FixSrcList can never fail.
                    Debugger.Break();
                pTab = sqlite3LocateTable(pParse, 0, pTblName.a[0].zName,
                pTblName.a[0].zDatabase);
                if (pTab == null)
                    goto exit_create_index;
                Debug.Assert(db.aDb[iDb].pSchema == pTab.pSchema);
            }
            else
            {
                Debug.Assert(pName == null);
                pTab = pParse.pNewTable;
                if (pTab == null)
                    goto exit_create_index;
                iDb = sqlite3SchemaToIndex(db, pTab.pSchema);
            }
            pDb = db.aDb[iDb];
            Debug.Assert(pTab != null);
            Debug.Assert(pParse.nErr == 0);
            if (pTab.zName.StartsWith("sqlite_", System.StringComparison.InvariantCultureIgnoreCase) && !pTab.zName.StartsWith("sqlite_altertab_", System.StringComparison.InvariantCultureIgnoreCase))
            {
                sqlite3ErrorMsg(pParse, "table %s may not be indexed", pTab.zName);
                goto exit_create_index;
            }
#if !SQLITE_OMIT_VIEW
            if (pTab.pSelect != null)
            {
                sqlite3ErrorMsg(pParse, "views may not be indexed");
                goto exit_create_index;
            }
#endif
            if (IsVirtual(pTab))
            {
                sqlite3ErrorMsg(pParse, "virtual tables may not be indexed");
                goto exit_create_index;
            }
            // Find the name of the index.  Make sure there is not already another index or table with the same name.
            // Exception:  If we are reading the names of permanent indices from the sqlite_master table (because some other process changed the schema) and
            // one of the index names collides with the name of a temporary table or index, then we will continue to process this index.
            // If pName==0 it means that we are dealing with a primary key or UNIQUE constraint.  We have to invent our own name.
            if (pName != null)
            {
                zName = sqlite3NameFromToken(db, pName);
                if (zName == null)
                    goto exit_create_index;
                if (SQLITE_OK != sqlite3CheckObjectName(pParse, zName))
                    goto exit_create_index;
                if (0 == db.init.busy)
                    if (sqlite3FindTable(db, zName, null) != null)
                    {
                        sqlite3ErrorMsg(pParse, "there is already a table named %s", zName);
                        goto exit_create_index;
                    }
                if (sqlite3FindIndex(db, zName, pDb.zName) != null)
                {
                    if (ifNotExist == 0)
                        sqlite3ErrorMsg(pParse, "index %s already exists", zName);
                    else
                    {
                        Debug.Assert(0 == db.init.busy);
                        sqlite3CodeVerifySchema(pParse, iDb);
                    }
                    goto exit_create_index;
                }
            }
            else
            {
                int n = 0;
                for (var pLoop = pTab.pIndex, n = 1; pLoop != null; pLoop = pLoop.pNext, n++) { }
                zName = sqlite3MPrintf(db, "sqlite_autoindex_%s_%d", pTab.zName, n);
                if (zName == null)
                    goto exit_create_index;
            }
            // Check for authorization to create an index.
            var zDb = pDb.zName;
            if (sqlite3AuthCheck(pParse, SQLITE_INSERT, SCHEMA_TABLE(iDb), 0, zDb))
                goto exit_create_index;
            var code = (OMIT_TEMPDB == 0 && iDb == 1 ? SQLITE_CREATE_TEMP_INDEX : SQLITE_CREATE_INDEX);
            if (sqlite3AuthCheck(pParse, i, zName, pTab.zName, zDb))
                goto exit_create_index;
            // If pList==0, it means this routine was called to make a primary key out of the last column added to the table under construction.
            // So create a fake list to simulate this.
            if (pList == null)
            {
                nullId.z = pTab.aCol[pTab.nCol - 1].zName;
                nullId.n = sqlite3Strlen30(nullId.z);
                pList = sqlite3ExprListAppend(pParse, null, null);
                if (pList == null)
                    goto exit_create_index;
                sqlite3ExprListSetName(pParse, pList, nullId, 0);
                pList.a[0].sortOrder = (u8)sortOrder;
            }
            // Figure out how many bytes of space are required to store explicitly specified collation sequence names.
            for (var i = 0; i < pList.nExpr; i++)
            {
                var pExpr = pList.a[i].pExpr;
                if (pExpr != null)
                {
                    var pColl = pExpr.pColl;
                    // Either pColl!=0 or there was an OOM failure.  But if an OOM failure we have quit before reaching this point. */
                    if (Check.ALWAYS(pColl != null))
                        nExtra += (1 + sqlite3Strlen30(pColl.zName));
                }
            }
            // Allocate the index structure.
            nName = sqlite3Strlen30(zName);
            nCol = pList.nExpr;
            pIndex = new Index();
            pIndex.azColl = new string[nCol + 1];
            pIndex.aiColumn = new int[nCol + 1];
            pIndex.aiRowEst = new int[nCol + 1];
            pIndex.aSortOrder = new byte[nCol + 1];
            zExtra = new StringBuilder(nName + 1);
            if (zName.Length == nName)
                pIndex.zName = zName;
            else
                pIndex.zName = zName.Substring(0, nName);
            pIndex.pTable = pTab;
            pIndex.nColumn = pList.nExpr;
            pIndex.onError = (byte)onError;
            pIndex.autoIndex = (byte)(pName == null ? 1 : 0);
            pIndex.pSchema = db.aDb[iDb].pSchema;
            Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
            // Check to see if we should honor DESC requests on index columns
            sortOrderMask = (pDb.pSchema.file_format >= 4 ? 1 : 0);
            // Scan the names of the columns of the table to be indexed and load the column indices into the Index structure.  Report an error
            // if any column is not found.
            for (var i = 0; i < pList.nExpr; i++)
            {
                int j;
                var zColName = pListItem.zName;
                for (j = 0; j < pTab.nCol; j++)
                {
                    var pTabCol = pTab.aCol[j];
                    if (zColName.Equals(pTabCol.zName, StringComparison.InvariantCultureIgnoreCase))
                        break;
                }
                if (j >= pTab.nCol)
                {
                    sqlite3ErrorMsg(pParse, "table %s has no column named %s",
                    pTab.zName, zColName);
                    pParse.checkSchema = 1;
                    goto exit_create_index;
                }
                pIndex.aiColumn[i] = j;
                // Justification of the ALWAYS(pListItem->pExpr->pColl):  Because of the way the "idxlist" non-terminal is constructed by the parser,
                // if pListItem->pExpr is not null then either pListItem->pExpr->pColl must exist or else there must have been an OOM error.  But if there
                // was an OOM error, we would never reach this point.
                string zColl;
                pListItem = pList.a[i];
                if (pListItem.pExpr != null && Check.ALWAYS(pListItem.pExpr.pColl))
                {
                    zColl = pListItem.pExpr.pColl.zName;
                    var nColl = sqlite3Strlen30(zColl);
                    Debug.Assert(nExtra >= nColl);
                    zExtra = new StringBuilder(zColl.Substring(0, nColl));// memcpy( zExtra, zColl, nColl );
                    zColl = zExtra.ToString();
                    nExtra -= nColl;
                }
                else
                {
                    zColl = pTab.aCol[j].zColl;
                    if (zColl == null)
                        zColl = db.pDfltColl.zName;
                }
                if (0 == db.init.busy && sqlite3LocateCollSeq(pParse, zColl) == null)
                    goto exit_create_index;
                pIndex.azColl[i] = zColl;
                var requestedSortOrder = (byte)((pListItem.sortOrder & sortOrderMask) != 0 ? 1 : 0);
                pIndex.aSortOrder[i] = (byte)requestedSortOrder;
            }
            sqlite3DefaultRowEst(pIndex);

            if (pTab == pParse.pNewTable)
            {
                // This routine has been called to create an automatic index as a result of a PRIMARY KEY or UNIQUE clause on a column definition, or
                // a PRIMARY KEY or UNIQUE clause following the column definitions. i.e. one of:
                // CREATE TABLE t(x PRIMARY KEY, y);
                // CREATE TABLE t(x, y, UNIQUE(x, y));
                //
                // Either way, check to see if the table already has such an index. If
                // so, don't bother creating this one. This only applies to
                // automatically created indices. Users can do as they wish with
                // explicit indices.
                //
                // Two UNIQUE or PRIMARY KEY constraints are considered equivalent
                // (and thus suppressing the second one) even if they have different
                // sort orders.
                //
                // If there are different collating sequences or if the columns of
                // the constraint occur in different orders, then the constraints are
                // considered distinct and both result in separate indices.
                Index pIdx;
                for (pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext)
                {
                    int k;
                    Debug.Assert(pIdx.onError != OE_None);
                    Debug.Assert(pIdx.autoIndex != 0);
                    Debug.Assert(pIndex.onError != OE_None);

                    if (pIdx.nColumn != pIndex.nColumn)
                        continue;
                    for (k = 0; k < pIdx.nColumn; k++)
                    {
                        string z1;
                        string z2;
                        if (pIdx.aiColumn[k] != pIndex.aiColumn[k])
                            break;
                        z1 = pIdx.azColl[k];
                        z2 = pIndex.azColl[k];
                        if (z1 != z2 && !z1.Equals(z2, StringComparison.InvariantCultureIgnoreCase))
                            break;
                    }
                    if (k == pIdx.nColumn)
                    {
                        if (pIdx.onError != pIndex.onError)
                        {
                            /* This constraint creates the same index as a previous
                            ** constraint specified somewhere in the CREATE TABLE statement.
                            ** However the ON CONFLICT clauses are different. If both this
                            ** constraint and the previous equivalent constraint have explicit
                            ** ON CONFLICT clauses this is an error. Otherwise, use the
                            ** explicitly specified behavior for the index.
                            */
                            if (!(pIdx.onError == OE_Default || pIndex.onError == OE_Default))
                            {
                                sqlite3ErrorMsg(pParse,
                                "conflicting ON CONFLICT clauses specified", 0);
                            }
                            if (pIdx.onError == OE_Default)
                            {
                                pIdx.onError = pIndex.onError;
                            }
                        }
                        goto exit_create_index;
                    }
                }
            }

            /* Link the new Index structure to its table and to the other
            ** in-memory database structures.
            */
            if (db.init.busy != 0)
            {
                Index p;
                Debug.Assert(sqlite3SchemaMutexHeld(db, 0, pIndex.pSchema));
                p = sqlite3HashInsert(ref pIndex.pSchema.idxHash,
                pIndex.zName, sqlite3Strlen30(pIndex.zName),
                pIndex);
                if (p != null)
                {
                    Debug.Assert(p == pIndex);  /* Malloc must have failed */
                    //        db.mallocFailed = 1;
                    goto exit_create_index;
                }
                db.flags |= SQLITE_InternChanges;
                if (pTblName != null)
                {
                    pIndex.tnum = db.init.newTnum;
                }
            }

                /* If the db.init.busy is 0 then create the index on disk.  This
                ** involves writing the index into the master table and filling in the
                ** index with the current table contents.
                **
                ** The db.init.busy is 0 when the user first enters a CREATE INDEX
                ** command.  db.init.busy is 1 when a database is opened and
                ** CREATE INDEX statements are read out of the master table.  In
                ** the latter case the index already exists on disk, which is why
                ** we don't want to recreate it.
                **
                ** If pTblName==0 it means this index is generated as a primary key
                ** or UNIQUE constraint of a CREATE TABLE statement.  Since the table
                ** has just been created, it contains no data and the index initialization
                ** step can be skipped.
                */
            else //if ( 0 == db.init.busy )
            {
                Vdbe v;
                string zStmt;
                int iMem = ++pParse.nMem;

                v = sqlite3GetVdbe(pParse);
                if (v == null)
                    goto exit_create_index;


                /* Create the rootpage for the index
                */
                sqlite3BeginWriteOperation(pParse, 1, iDb);
                sqlite3VdbeAddOp2(v, OP_CreateIndex, iDb, iMem);

                /* Gather the complete text of the CREATE INDEX statement into
                ** the zStmt variable
                */
                if (pStart != null)
                {
                    Debug.Assert(pEnd != null);
                    /* A named index with an explicit CREATE INDEX statement */
                    zStmt = sqlite3MPrintf(db, "CREATE%s INDEX %.*s",
                    onError == OE_None ? "" : " UNIQUE",
                     (int)(pName.z.Length - pEnd.z.Length) + 1,
                    pName.z);
                }
                else
                {
                    /* An automatic index created by a PRIMARY KEY or UNIQUE constraint */
                    /* zStmt = sqlite3MPrintf(""); */
                    zStmt = null;
                }

                /* Add an entry in sqlite_master for this index
                */
                sqlite3NestedParse(pParse,
                "INSERT INTO %Q.%s VALUES('index',%Q,%Q,#%d,%Q);",
                db.aDb[iDb].zName, SCHEMA_TABLE(iDb),
                pIndex.zName,
                pTab.zName,
                iMem,
                zStmt
                );
                sqlite3DbFree(db, ref zStmt);

                /* Fill the index with data and reparse the schema. Code an OP_Expire
                ** to invalidate all pre-compiled statements.
                */
                if (pTblName != null)
                {
                    sqlite3RefillIndex(pParse, pIndex, iMem);
                    sqlite3ChangeCookie(pParse, iDb);
                    sqlite3VdbeAddParseSchemaOp(v, iDb,
                      sqlite3MPrintf(db, "name='%q' AND type='index'", pIndex.zName));
                    sqlite3VdbeAddOp1(v, OP_Expire, 0);
                }
            }

            /* When adding an index to the list of indices for a table, make
            ** sure all indices labeled OE_Replace come after all those labeled
            ** OE_Ignore.  This is necessary for the correct constraint check
            ** processing (in sqlite3GenerateConstraintChecks()) as part of
            ** UPDATE and INSERT statements.
            */
            if (db.init.busy != 0 || pTblName == null)
            {
                if (onError != OE_Replace || pTab.pIndex == null
                || pTab.pIndex.onError == OE_Replace)
                {
                    pIndex.pNext = pTab.pIndex;
                    pTab.pIndex = pIndex;
                }
                else
                {
                    Index pOther = pTab.pIndex;
                    while (pOther.pNext != null && pOther.pNext.onError != OE_Replace)
                    {
                        pOther = pOther.pNext;
                    }
                    pIndex.pNext = pOther.pNext;
                    pOther.pNext = pIndex;
                }
                pRet = pIndex;
                pIndex = null;
            }

            /* Clean up before exiting */
        exit_create_index:
            if (pIndex != null)
            {
                //sqlite3DbFree(db, ref pIndex.zColAff );
                sqlite3DbFree(db, ref pIndex);
            }
            sqlite3ExprListDelete(db, ref pList);
            sqlite3SrcListDelete(db, ref pTblName);
            sqlite3DbFree(db, ref zName);
            return pRet;
        }

        /*
        ** Fill the Index.aiRowEst[] array with default information - information
        ** to be used when we have not run the ANALYZE command.
        **
        ** aiRowEst[0] is suppose to contain the number of elements in the index.
        ** Since we do not know, guess 1 million.  aiRowEst[1] is an estimate of the
        ** number of rows in the table that match any particular value of the
        ** first column of the index.  aiRowEst[2] is an estimate of the number
        ** of rows that match any particular combiniation of the first 2 columns
        ** of the index.  And so forth.  It must always be the case that
        *
        **           aiRowEst[N]<=aiRowEst[N-1]
        **           aiRowEst[N]>=1
        **
        ** Apart from that, we have little to go on besides intuition as to
        ** how aiRowEst[] should be initialized.  The numbers generated here
        ** are based on typical values found in actual indices.
        */
        static void sqlite3DefaultRowEst(Index pIdx)
        {
            int[] a = pIdx.aiRowEst;
            int i;
            int n;
            Debug.Assert(a != null);
            a[0] = (int)pIdx.pTable.nRowEst;
            if (a[0] < 10)
                a[0] = 10;
            n = 10;
            for (i = 1; i <= pIdx.nColumn; i++)
            {
                a[i] = n;
                if (n > 5)
                    n--;
            }
            if (pIdx.onError != OE_None)
            {
                a[pIdx.nColumn] = 1;
            }
        }

    }
}