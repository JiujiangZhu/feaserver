using System.Diagnostics;
namespace FeaServer.Engine.Query
{
    public class Build
    {
        // Reclaim the memory used by an index
        internal static void freeIndex(sqlite3 db, ref Index p)
        {
#if !SQLITE_OMIT_ANALYZE
            sqlite3DeleteIndexSamples(db, p);
#endif
            sqlite3DbFree(db, ref p.zColAff);
            sqlite3DbFree(db, ref p);
        }

        // For the index called zIdxName which is found in the database iDb, unlike that index from its Table then remove the index from
        // the index hash table and free all memory structures associated with the index.
        internal static void sqlite3UnlinkAndDeleteIndex(sqlite3 db, int iDb, string zIdxName)
        {
            Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
            var pHash = db.aDb[iDb].pSchema.idxHash;
            var len = sqlite3Strlen30(zIdxName);
            var pIndex = sqlite3HashInsert(ref pHash, zIdxName, len, (Index)null);
            if (Check.ALWAYS(pIndex))
            {
                if (pIndex.pTable.pIndex == pIndex)
                    pIndex.pTable.pIndex = pIndex.pNext;
                else
                {
                    // Justification of ALWAYS();  The index must be on the list of indices.
                    var p = pIndex.pTable.pIndex;
                    while (Check.ALWAYS(p != null) && p.pNext != pIndex)
                        p = p.pNext;
                    if (Check.ALWAYS(p != null && p.pNext == pIndex))
                        p.pNext = pIndex.pNext;
                }
                freeIndex(db, ref pIndex);
            }
            db.flags |= SQLITE_InternChanges;
        }

        // Delete memory allocated for the column names of a table or view (the Table.aCol[] array).
        internal static void sqliteDeleteColumnNames(sqlite3 db, Table pTable)
        {
            Debug.Assert(pTable != null);
            for (var i = 0; i < pTable.nCol; i++)
            {
                var pCol = pTable.aCol[i];
                if (pCol != null)
                {
                    sqlite3DbFree(db, ref pCol.zName);
                    sqlite3ExprDelete(db, ref pCol.pDflt);
                    sqlite3DbFree(db, ref pCol.zDflt);
                    sqlite3DbFree(db, ref pCol.zType);
                    sqlite3DbFree(db, ref pCol.zColl);
                }
            }
        }

        // Remove the memory data structures associated with the given Table.  No changes are made to disk by this routine.
        //
        // This routine just deletes the data structure.  It does not unlink the table data structure from the hash table.  But it does destroy
        // memory structures of the indices and foreign keys associated with the table.
        internal static void sqlite3DeleteTable(sqlite3 db, ref Table pTable)
        {
            Debug.Assert(null == pTable || pTable.nRef > 0);
            // Do not delete the table until the reference count reaches zero.
            if (null == pTable)
                return;
            if ((// ( !db || db->pnBytesFreed == 0 ) && 
              (--pTable.nRef) > 0))
                return;
            // Delete all indices associated with this table.
            Index pNext;
            for (var pIndex = pTable.pIndex; pIndex != null; pIndex = pNext)
            {
                pNext = pIndex.pNext;
                Debug.Assert(pIndex.pSchema == pTable.pSchema);
                var zName = pIndex.zName;
#if !DEBUG
                var pOld = sqlite3HashInsert(ref pIndex.pSchema.idxHash, zName, sqlite3Strlen30(zName), (Index)null);
                Debug.Assert(db == null || sqlite3SchemaMutexHeld(db, 0, pIndex.pSchema));
                Debug.Assert(pOld == pIndex || pOld == null);
#else
                sqlite3HashInsert(ref pIndex.pSchema.idxHash, zName, sqlite3Strlen30(zName), (Index)null);
#endif
                freeIndex(db, ref pIndex);
            }
            // Delete any foreign keys attached to this table.
            sqlite3FkDelete(db, pTable);

            // Delete the Table structure itself.
            sqliteDeleteColumnNames(db, pTable);
            sqlite3DbFree(db, ref pTable.zName);
            sqlite3DbFree(db, ref pTable.zColAff);
            sqlite3SelectDelete(db, ref pTable.pSelect);
#if !SQLITE_OMIT_CHECK
            sqlite3ExprDelete(db, ref pTable.pCheck);
#endif
#if !SQLITE_OMIT_VIRTUALTABLE
            sqlite3VtabClear(db, pTable);
#endif
            pTable = null;
        }

        // Unlink the given table from the hash tables and the delete the table structure with all its indices and foreign keys.
        internal static void sqlite3UnlinkAndDeleteTable(sqlite3 db, int iDb, string zTabName)
        {
            Debug.Assert(db != null);
            Debug.Assert(iDb >= 0 && iDb < db.nDb);
            Debug.Assert(zTabName != null);
            Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
            //testcase(zTabName.Length == 0);  /* Zero-length table names are allowed */
            var pDb = db.aDb[iDb];
            var p = sqlite3HashInsert(ref pDb.pSchema.tblHash, zTabName,
            sqlite3Strlen30(zTabName), (Table)null);
            sqlite3DeleteTable(db, ref p);
            db.flags |= SQLITE_InternChanges;
        }

        // Write code to erase the table with root-page iTable from database iDb. Also write code to modify the sqlite_master table and internal schema
        // if a root-page of another table is moved by the btree-layer whilst erasing iTable (this can happen with an auto-vacuum database).
        internal static void destroyRootPage(Parse pParse, int iTable, int iDb)
        {
            var v = sqlite3GetVdbe(pParse);
            var r1 = sqlite3GetTempReg(pParse);
            sqlite3VdbeAddOp3(v, OP_Destroy, iTable, r1, iDb);
            sqlite3MayAbort(pParse);
#if !SQLITE_OMIT_AUTOVACUUM
            // OP_Destroy stores an in integer r1. If this integer is non-zero, then it is the root page number of a table moved to
            // location iTable. The following code modifies the sqlite_master table to reflect this.
            //
            // The "#NNN" in the SQL is a special constant that means whatever value is in register NNN.  See grammar rules associated with the TK_REGISTER
            // token for additional information.
            sqlite3NestedParse(pParse, "UPDATE %Q.%s SET rootpage=%d WHERE #%d AND rootpage=#%d", pParse.db.aDb[iDb].zName, SCHEMA_TABLE(iDb), iTable, r1, r1);
#endif
            sqlite3ReleaseTempReg(pParse, r1);
        }

        // Write VDBE code to erase table pTab and all associated indices on disk. Code to update the sqlite_master tables and internal schema definitions
        // in case a root-page belonging to another table is moved by the btree layer is also added (this can happen with an auto-vacuum database).
        internal static void destroyTable(Parse pParse, Table pTab)
        {
#if SQLITE_OMIT_AUTOVACUUM
            var iDb = sqlite3SchemaToIndex(pParse.db, pTab.pSchema);
            destroyRootPage(pParse, pTab.tnum, iDb);
            for (var pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext)
                destroyRootPage(pParse, pIdx.tnum, iDb);
#else
            // If the database may be auto-vacuum capable (if SQLITE_OMIT_AUTOVACUUM is not defined), then it is important to call OP_Destroy on the
            // table and index root-pages in order, starting with the numerically largest root-page number. This guarantees that none of the root-pages
            // to be destroyed is relocated by an earlier OP_Destroy. i.e. if the following were coded:
            // OP_Destroy 4 0
            // ...
            // OP_Destroy 5 0
            // and root page 5 happened to be the largest root-page number in the database, then root page 5 would be moved to page 4 by the
            // "OP_Destroy 4 0" opcode. The subsequent "OP_Destroy 5 0" would hit a free-list page.
            var iTab = pTab.tnum;
            var iDestroyed = 0;
            while (true)
            {
                var iLargest = 0;
                if (iDestroyed == 0 || iTab < iDestroyed)
                    iLargest = iTab;
                for (var pIdx = pTab.pIndex; pIdx != null; pIdx = pIdx.pNext)
                {
                    var iIdx = pIdx.tnum;
                    Debug.Assert(pIdx.pSchema == pTab.pSchema);
                    if ((iDestroyed == 0 || (iIdx < iDestroyed)) && iIdx > iLargest)
                        iLargest = iIdx;
                }
                if (iLargest == 0)
                    return;
                else
                {
                    var iDb = sqlite3SchemaToIndex(pParse.db, pTab.pSchema);
                    destroyRootPage(pParse, iLargest, iDb);
                    iDestroyed = iLargest;
                }
            }
#endif
        }

        // This routine is called to do the work of a DROP TABLE statement. pName is the name of the table to be dropped.
        internal static void sqlite3DropTable(Parse pParse, SrcList pName, int isView, int noErr)
        {
            var db = pParse.db;
            Debug.Assert(pParse.nErr == 0);
            Debug.Assert(pName.nSrc == 1);
            if (noErr != 0)
                db.suppressErr++;
            var pTab = sqlite3LocateTable(pParse, isView,
            pName.a[0].zName, pName.a[0].zDatabase);
            if (noErr != 0)
                db.suppressErr--;
            if (pTab == null)
            {
                if (noErr != 0)
                    sqlite3CodeVerifyNamedSchema(pParse, pName.a[0].zDatabase);
                goto exit_drop_table;
            }
            var iDb = sqlite3SchemaToIndex(db, pTab.pSchema);
            Debug.Assert(iDb >= 0 && iDb < db.nDb);
            // If pTab is a virtual table, call ViewGetColumnNames() to ensure it is initialized.
            if (IsVirtual(pTab) && sqlite3ViewGetColumnNames(pParse, pTab) != 0)
                goto exit_drop_table;
            var zTab = SCHEMA_TABLE(iDb);
            var zDb = db.aDb[iDb].zName;
            string zArg2 = null;
            if (sqlite3AuthCheck(pParse, SQLITE_DELETE, zTab, 0, zDb))
                goto exit_drop_table;
            int code;
            if (isView)
                code = (OMIT_TEMPDB == 0 && iDb == 1 ? SQLITE_DROP_TEMP_VIEW : SQLITE_DROP_VIEW);
            else if (IsVirtual(pTab))
            {
                code = SQLITE_DROP_VTABLE;
                zArg2 = sqlite3GetVTable(db, pTab)->pMod->zName;
            }
            else
                code = (OMIT_TEMPDB == 0 && iDb == 1 ? SQLITE_DROP_TEMP_TABLE : SQLITE_DROP_TABLE);
            if (sqlite3AuthCheck(pParse, code, pTab.zName, zArg2, zDb))
                goto exit_drop_table;
            if (sqlite3AuthCheck(pParse, SQLITE_DELETE, pTab.zName, 0, zDb))
                goto exit_drop_table;
            if (pTab.zName.StartsWith("sqlite_", System.StringComparison.InvariantCultureIgnoreCase))
            {
                sqlite3ErrorMsg(pParse, "table %s may not be dropped", pTab.zName);
                goto exit_drop_table;
            }

#if !SQLITE_OMIT_VIEW
            // Ensure DROP TABLE is not used on a view, and DROP VIEW is not used on a table.
            if (isView != 0 && pTab.pSelect == null)
            {
                sqlite3ErrorMsg(pParse, "use DROP TABLE to delete table %s", pTab.zName);
                goto exit_drop_table;
            }
            if (0 == isView && pTab.pSelect != null)
            {
                sqlite3ErrorMsg(pParse, "use DROP VIEW to delete view %s", pTab.zName);
                goto exit_drop_table;
            }
#endif
            // Generate code to remove the table from the master table on disk.
            var v = sqlite3GetVdbe(pParse);
            if (v != null)
            {
                var pDb = db.aDb[iDb];
                sqlite3BeginWriteOperation(pParse, 1, iDb);
#if !SQLITE_OMIT_VIRTUALTABLE
                if (IsVirtual(pTab))
                    sqlite3VdbeAddOp0(v, OP_VBegin);
#endif
                sqlite3FkDropTable(pParse, pName, pTab);
                // Drop all triggers associated with the table being dropped. Code is generated to remove entries from sqlite_master and/or
                // sqlite_temp_master if required.
                var pTrigger = sqlite3TriggerList(pParse, pTab);
                while (pTrigger != null)
                {
                    Debug.Assert(pTrigger.pSchema == pTab.pSchema ||
                    pTrigger.pSchema == db.aDb[1].pSchema);
                    sqlite3DropTriggerPtr(pParse, pTrigger);
                    pTrigger = pTrigger.pNext;
                }
#if !SQLITE_OMIT_AUTOINCREMENT
                // Remove any entries of the sqlite_sequence table associated with the table being dropped. This is done before the table is dropped
                // at the btree level, in case the sqlite_sequence table needs to move as a result of the drop (can happen in auto-vacuum mode).
                if ((pTab.tabFlags & TF_Autoincrement) != 0)
                    sqlite3NestedParse(pParse, "DELETE FROM %s.sqlite_sequence WHERE name=%Q", pDb.zName, pTab.zName);
#endif
                // Drop all SQLITE_MASTER table and index entries that refer to the table. The program name loops through the master table and deletes
                // every row that refers to a table of the same name as the one being dropped. Triggers are handled seperately because a trigger can be
                // created in the temp database that refers to a table in another database.
                sqlite3NestedParse(pParse, "DELETE FROM %Q.%s WHERE tbl_name=%Q and type!='trigger'", pDb.zName, SCHEMA_TABLE(iDb), pTab.zName);
                // Drop any statistics from the sqlite_stat1 table, if it exists
                if (sqlite3FindTable(db, "sqlite_stat1", db.aDb[iDb].zName) != null)
                    sqlite3NestedParse(pParse, "DELETE FROM %Q.sqlite_stat1 WHERE tbl=%Q", pDb.zName, pTab.zName);
                if (0 == isView && !IsVirtual(pTab))
                    destroyTable(pParse, pTab);
                // Remove the table entry from SQLite's internal schema and modify the schema cookie.
                if (IsVirtual(pTab))
                    sqlite3VdbeAddOp4(v, OP_VDestroy, iDb, 0, 0, pTab.zName, 0);
                sqlite3VdbeAddOp4(v, OP_DropTable, iDb, 0, 0, pTab.zName, 0);
                sqlite3ChangeCookie(pParse, iDb);
            }
            sqliteViewResetAll(db, iDb);
        exit_drop_table:
            sqlite3SrcListDelete(db, ref pName);
        }

        // This routine will drop an existing named index.  This routine implements the DROP INDEX statement.
        internal static void sqlite3DropIndex(Parse pParse, SrcList pName, int ifExists)
        {
            Debug.Assert(pParse.nErr == 0);   // Never called with prior errors
            Debug.Assert(pName.nSrc == 1);
            if (SQLITE_OK != sqlite3ReadSchema(pParse))
                goto exit_drop_index;
            var db = pParse.db;
            var pIndex = sqlite3FindIndex(db, pName.a[0].zName, pName.a[0].zDatabase);
            if (pIndex == null)
            {
                if (ifExists == 0)
                    sqlite3ErrorMsg(pParse, "no such index: %S", pName, 0);
                else
                    sqlite3CodeVerifyNamedSchema(pParse, pName.a[0].zDatabase);
                pParse.checkSchema = 1;
                goto exit_drop_index;
            }
            if (pIndex.autoIndex != 0)
            {
                sqlite3ErrorMsg(pParse, "index associated with UNIQUE " + "or PRIMARY KEY constraint cannot be dropped", 0);
                goto exit_drop_index;
            }
            var iDb = sqlite3SchemaToIndex(db, pIndex.pSchema);
                var pTab = pIndex.pTable;
                var zDb = db.aDb[iDb].zName;
                var zTab = SCHEMA_TABLE(iDb);
                if (sqlite3AuthCheck(pParse, SQLITE_DELETE, zTab, 0, zDb))
                    goto exit_drop_index;
                var code (OMIT_TEMPDB == 0 && iDb ? SQLITE_DROP_TEMP_INDEX : SQLITE_DROP_INDEX);
                if (sqlite3AuthCheck(pParse, code, pIndex.zName, pTab.zName, zDb))
                    goto exit_drop_index;
            // Generate code to remove the index and from the master table
            var v = sqlite3GetVdbe(pParse);
            if (v != null)
            {
                sqlite3BeginWriteOperation(pParse, 1, iDb);
                sqlite3NestedParse(pParse, "DELETE FROM %Q.%s WHERE name=%Q AND type='index'", db.aDb[iDb].zName, SCHEMA_TABLE(iDb), pIndex.zName );
                if (sqlite3FindTable(db, "sqlite_stat1", db.aDb[iDb].zName) != null)
                    sqlite3NestedParse(pParse, "DELETE FROM %Q.sqlite_stat1 WHERE idx=%Q", db.aDb[iDb].zName, pIndex.zName );
                sqlite3ChangeCookie(pParse, iDb);
                destroyRootPage(pParse, pIndex.tnum, iDb);
                sqlite3VdbeAddOp4(v, OP_DropIndex, iDb, 0, 0, pIndex.zName, 0);
            }
        exit_drop_index:
            sqlite3SrcListDelete(db, ref pName);
        }
    }
}