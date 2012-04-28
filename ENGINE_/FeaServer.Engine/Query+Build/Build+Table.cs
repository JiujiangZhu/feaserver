namespace FeaServer.Engine.Query
{
    public partial class Build
    {

        /*
        ** Begin constructing a new table representation in memory.  This is
        ** the first of several action routines that get called in response
        ** to a CREATE TABLE statement.  In particular, this routine is called
        ** after seeing tokens "CREATE" and "TABLE" and the table name. The isTemp
        ** flag is true if the table should be stored in the auxiliary database
        ** file instead of in the main database file.  This is normally the case
        ** when the "TEMP" or "TEMPORARY" keyword occurs in between
        ** CREATE and TABLE.
        **
        ** The new table record is initialized and put in pParse.pNewTable.
        ** As more of the CREATE TABLE statement is parsed, additional action
        ** routines will be called to add more information to this record.
        ** At the end of the CREATE TABLE statement, the sqlite3EndTable() routine
        ** is called to complete the construction of the new table record.
        */
        static void sqlite3StartTable(
        Parse pParse,   /* Parser context */
        Token pName1,   /* First part of the name of the table or view */
        Token pName2,   /* Second part of the name of the table or view */
        int isTemp,      /* True if this is a TEMP table */
        int isView,      /* True if this is a VIEW */
        int isVirtual,   /* True if this is a VIRTUAL table */
        int noErr        /* Do nothing if table already exists */
        )
        {
            Table pTable;
            string zName = null; /* The name of the new table */
            sqlite3 db = pParse.db;
            Vdbe v;
            int iDb;         /* Database number to create the table in */
            Token pName = new Token();    /* Unqualified name of the table to create */

            /* The table or view name to create is passed to this routine via tokens
            ** pName1 and pName2. If the table name was fully qualified, for example:
            **
            ** CREATE TABLE xxx.yyy (...);
            **
            ** Then pName1 is set to "xxx" and pName2 "yyy". On the other hand if
            ** the table name is not fully qualified, i.e.:
            **
            ** CREATE TABLE yyy(...);
            **
            ** Then pName1 is set to "yyy" and pName2 is "".
            **
            ** The call below sets the pName pointer to point at the token (pName1 or
            ** pName2) that stores the unqualified table name. The variable iDb is
            ** set to the index of the database that the table or view is to be
            ** created in.
            */
            iDb = sqlite3TwoPartName(pParse, pName1, pName2, ref pName);
            if (iDb < 0)
                return;
            if (0 == OMIT_TEMPDB && isTemp != 0 && pName2.n > 0 && iDb != 1)
            {
                /* If creating a temp table, the name may not be qualified. Unless 
                ** the database name is "temp" anyway.  */

                sqlite3ErrorMsg(pParse, "temporary table name must be unqualified");
                return;
            }
            if (OMIT_TEMPDB == 0 && isTemp != 0)
                iDb = 1;

            pParse.sNameToken = pName;
            zName = sqlite3NameFromToken(db, pName);
            if (zName == null)
                return;
            if (SQLITE_OK != sqlite3CheckObjectName(pParse, zName))
            {
                goto begin_table_error;
            }
            if (db.init.iDb == 1)
                isTemp = 1;
#if !SQLITE_OMIT_AUTHORIZATION
            Debug.Assert((isTemp & 1) == isTemp);
            {
                int code;
                string zDb = db.aDb[iDb].zName;
                if (sqlite3AuthCheck(pParse, SQLITE_INSERT, SCHEMA_TABLE(isTemp), 0, zDb))
                {
                    goto begin_table_error;
                }
                if (isView)
                {
                    if (OMIT_TEMPDB == 0 && isTemp)
                    {
                        code = SQLITE_CREATE_TEMP_VIEW;
                    }
                    else
                    {
                        code = SQLITE_CREATE_VIEW;
                    }
                }
                else
                {
                    if (OMIT_TEMPDB == 0 && isTemp)
                    {
                        code = SQLITE_CREATE_TEMP_TABLE;
                    }
                    else
                    {
                        code = SQLITE_CREATE_TABLE;
                    }
                }
                if (null == isVirtual && sqlite3AuthCheck(pParse, code, zName, 0, zDb))
                {
                    goto begin_table_error;
                }
            }
#endif

            /* Make sure the new table name does not collide with an existing
** index or table name in the same database.  Issue an error message if
** it does. The exception is if the statement being parsed was passed
** to an sqlite3_declare_vtab() call. In that case only the column names
** and types will be used, so there is no need to test for namespace
** collisions.
*/
            if (!IN_DECLARE_VTAB(pParse))
            {
                String zDb = db.aDb[iDb].zName;
                if (SQLITE_OK != sqlite3ReadSchema(pParse))
                {
                    goto begin_table_error;
                }
                pTable = sqlite3FindTable(db, zName, zDb);
                if (pTable != null)
                {
                    if (noErr == 0)
                    {
                        sqlite3ErrorMsg(pParse, "table %T already exists", pName);
                    }
                    else
                    {
                        Debug.Assert(0 == db.init.busy);
                        sqlite3CodeVerifySchema(pParse, iDb);
                    }
                    goto begin_table_error;
                }
                if (sqlite3FindIndex(db, zName, zDb) != null)
                {
                    sqlite3ErrorMsg(pParse, "there is already an index named %s", zName);
                    goto begin_table_error;
                }
            }

            pTable = new Table();// sqlite3DbMallocZero(db, Table).Length;
            //if ( pTable == null )
            //{
            //  db.mallocFailed = 1;
            //  pParse.rc = SQLITE_NOMEM;
            //  pParse.nErr++;
            //  goto begin_table_error;
            //}
            pTable.zName = zName;
            pTable.iPKey = -1;
            pTable.pSchema = db.aDb[iDb].pSchema;
            pTable.nRef = 1;
            pTable.nRowEst = 1000000;
            Debug.Assert(pParse.pNewTable == null);
            pParse.pNewTable = pTable;

            /* If this is the magic sqlite_sequence table used by autoincrement,
            ** then record a pointer to this table in the main database structure
            ** so that INSERT can find the table easily.
            */
#if !SQLITE_OMIT_AUTOINCREMENT
            if (pParse.nested == 0 && zName == "sqlite_sequence")
            {
                Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
                pTable.pSchema.pSeqTab = pTable;
            }
#endif

            /* Begin generating the code that will insert the table record into
** the SQLITE_MASTER table.  Note in particular that we must go ahead
** and allocate the record number for the table entry now.  Before any
** PRIMARY KEY or UNIQUE keywords are parsed.  Those keywords will cause
** indices to be created and the table record must come before the
** indices.  Hence, the record number for the table must be allocated
** now.
*/
            if (0 == db.init.busy && (v = sqlite3GetVdbe(pParse)) != null)
            {
                int j1;
                int fileFormat;
                int reg1, reg2, reg3;
                sqlite3BeginWriteOperation(pParse, 0, iDb);

                if (isVirtual != 0)
                {
                    sqlite3VdbeAddOp0(v, OP_VBegin);
                }

                /* If the file format and encoding in the database have not been set,
                ** set them now.
                */
                reg1 = pParse.regRowid = ++pParse.nMem;
                reg2 = pParse.regRoot = ++pParse.nMem;
                reg3 = ++pParse.nMem;
                sqlite3VdbeAddOp3(v, OP_ReadCookie, iDb, reg3, BTREE_FILE_FORMAT);
                sqlite3VdbeUsesBtree(v, iDb);
                j1 = sqlite3VdbeAddOp1(v, OP_If, reg3);
                fileFormat = (db.flags & SQLITE_LegacyFileFmt) != 0 ?
                1 : SQLITE_MAX_FILE_FORMAT;
                sqlite3VdbeAddOp2(v, OP_Integer, fileFormat, reg3);
                sqlite3VdbeAddOp3(v, OP_SetCookie, iDb, BTREE_FILE_FORMAT, reg3);
                sqlite3VdbeAddOp2(v, OP_Integer, ENC(db), reg3);
                sqlite3VdbeAddOp3(v, OP_SetCookie, iDb, BTREE_TEXT_ENCODING, reg3);
                sqlite3VdbeJumpHere(v, j1);

                /* This just creates a place-holder record in the sqlite_master table.
                ** The record created does not contain anything yet.  It will be replaced
                ** by the real entry in code generated at sqlite3EndTable().
                **
                ** The rowid for the new entry is left in register pParse->regRowid.
                ** The root page number of the new table is left in reg pParse->regRoot.
                ** The rowid and root page number values are needed by the code that
                ** sqlite3EndTable will generate.
                */
                if (isView != 0 || isVirtual != 0)
                {
                    sqlite3VdbeAddOp2(v, OP_Integer, 0, reg2);
                }
                else
                {
                    sqlite3VdbeAddOp2(v, OP_CreateTable, iDb, reg2);
                }
                sqlite3OpenMasterTable(pParse, iDb);
                sqlite3VdbeAddOp2(v, OP_NewRowid, 0, reg1);
                sqlite3VdbeAddOp2(v, OP_Null, 0, reg3);
                sqlite3VdbeAddOp3(v, OP_Insert, 0, reg3, reg1);
                sqlite3VdbeChangeP5(v, OPFLAG_APPEND);
                sqlite3VdbeAddOp0(v, OP_Close);
            }

            /* Normal (non-error) return. */
            return;

          /* If an error occurs, we jump here */
        begin_table_error:
            sqlite3DbFree(db, ref zName);
            return;
        }

        /*
        ** This macro is used to compare two strings in a case-insensitive manner.
        ** It is slightly faster than calling sqlite3StrICmp() directly, but
        ** produces larger code.
        **
        ** WARNING: This macro is not compatible with the strcmp() family. It
        ** returns true if the two strings are equal, otherwise false.
        */
        //#define STRICMP(x, y) (\
        //sqlite3UpperToLower[*(unsigned char )(x)]==   \
        //sqlite3UpperToLower[*(unsigned char )(y)]     \
        //&& sqlite3StrICmp((x)+1,(y)+1)==0 )

        /*
        ** Add a new column to the table currently being constructed.
        **
        ** The parser calls this routine once for each column declaration
        ** in a CREATE TABLE statement.  sqlite3StartTable() gets called
        ** first to get things going.  Then this routine is called for each
        ** column.
        */
        static void sqlite3AddColumn(Parse pParse, Token pName)
        {
            Table p;
            int i;
            string z;
            Column pCol;
            sqlite3 db = pParse.db;
            if ((p = pParse.pNewTable) == null)
                return;
#if SQLITE_MAX_COLUMN || !SQLITE_MAX_COLUMN
            if (p.nCol + 1 > db.aLimit[SQLITE_LIMIT_COLUMN])
            {
                sqlite3ErrorMsg(pParse, "too many columns on %s", p.zName);
                return;
            }
#endif
            z = sqlite3NameFromToken(db, pName);
            if (z == null)
                return;
            for (i = 0; i < p.nCol; i++)
            {
                if (z.Equals(p.aCol[i].zName, StringComparison.InvariantCultureIgnoreCase))
                {//STRICMP(z, p.aCol[i].zName) ){
                    sqlite3ErrorMsg(pParse, "duplicate column name: %s", z);
                    sqlite3DbFree(db, ref z);
                    return;
                }
            }
            if ((p.nCol & 0x7) == 0)
            {
                //aNew = sqlite3DbRealloc(db,p.aCol,(p.nCol+8)*sizeof(p.aCol[0]));
                //if( aNew==0 ){
                //  sqlite3DbFree(db,ref z);
                //  return;
                //}
                Array.Resize(ref p.aCol, p.nCol + 8);
            }
            p.aCol[p.nCol] = new Column();
            pCol = p.aCol[p.nCol];
            //memset(pCol, 0, sizeof(p.aCol[0]));
            pCol.zName = z;

            /* If there is no type specified, columns have the default affinity
            ** 'NONE'. If there is a type specified, then sqlite3AddColumnType() will
            ** be called next to set pCol.affinity correctly.
            */
            pCol.affinity = SQLITE_AFF_NONE;
            p.nCol++;
        }

        /*
        ** This routine is called by the parser while in the middle of
        ** parsing a CREATE TABLE statement.  A "NOT NULL" constraint has
        ** been seen on a column.  This routine sets the notNull flag on
        ** the column currently under construction.
        */
        static void sqlite3AddNotNull(Parse pParse, int onError)
        {
            Table p;
            p = pParse.pNewTable;
            if (p == null || NEVER(p.nCol < 1))
                return;
            p.aCol[p.nCol - 1].notNull = (u8)onError;
        }

        /*
        ** Scan the column type name zType (length nType) and return the
        ** associated affinity type.
        **
        ** This routine does a case-independent search of zType for the
        ** substrings in the following table. If one of the substrings is
        ** found, the corresponding affinity is returned. If zType contains
        ** more than one of the substrings, entries toward the top of
        ** the table take priority. For example, if zType is 'BLOBINT',
        ** SQLITE_AFF_INTEGER is returned.
        **
        ** Substring     | Affinity
        ** --------------------------------
        ** 'INT'         | SQLITE_AFF_INTEGER
        ** 'CHAR'        | SQLITE_AFF_TEXT
        ** 'CLOB'        | SQLITE_AFF_TEXT
        ** 'TEXT'        | SQLITE_AFF_TEXT
        ** 'BLOB'        | SQLITE_AFF_NONE
        ** 'REAL'        | SQLITE_AFF_REAL
        ** 'FLOA'        | SQLITE_AFF_REAL
        ** 'DOUB'        | SQLITE_AFF_REAL
        **
        ** If none of the substrings in the above table are found,
        ** SQLITE_AFF_NUMERIC is returned.
        */
        static char sqlite3AffinityType(string zIn)
        {
            //u32 h = 0;
            //char aff = SQLITE_AFF_NUMERIC;
            zIn = zIn.ToLower();
            if (zIn.Contains("char") || zIn.Contains("clob") || zIn.Contains("text"))
                return SQLITE_AFF_TEXT;
            if (zIn.Contains("blob"))
                return SQLITE_AFF_NONE;
            if (zIn.Contains("doub") || zIn.Contains("floa") || zIn.Contains("real"))
                return SQLITE_AFF_REAL;
            if (zIn.Contains("int"))
                return SQLITE_AFF_INTEGER;
            return SQLITE_AFF_NUMERIC;
            //      string zEnd = pType.z.Substring(pType.n);

            //      while( zIn!=zEnd ){
            //        h = (h<<8) + sqlite3UpperToLower[*zIn];
            //        zIn++;
            //        if( h==(('c'<<24)+('h'<<16)+('a'<<8)+'r') ){             /* CHAR */
            //          aff = SQLITE_AFF_TEXT;
            //        }else if( h==(('c'<<24)+('l'<<16)+('o'<<8)+'b') ){       /* CLOB */
            //          aff = SQLITE_AFF_TEXT;
            //        }else if( h==(('t'<<24)+('e'<<16)+('x'<<8)+'t') ){       /* TEXT */
            //          aff = SQLITE_AFF_TEXT;
            //        }else if( h==(('b'<<24)+('l'<<16)+('o'<<8)+'b')          /* BLOB */
            //            && (aff==SQLITE_AFF_NUMERIC || aff==SQLITE_AFF_REAL) ){
            //          aff = SQLITE_AFF_NONE;
            //#if !SQLITE_OMIT_FLOATING_POINT
            //        }else if( h==(('r'<<24)+('e'<<16)+('a'<<8)+'l')          /* REAL */
            //            && aff==SQLITE_AFF_NUMERIC ){
            //          aff = SQLITE_AFF_REAL;
            //        }else if( h==(('f'<<24)+('l'<<16)+('o'<<8)+'a')          /* FLOA */
            //            && aff==SQLITE_AFF_NUMERIC ){
            //          aff = SQLITE_AFF_REAL;
            //        }else if( h==(('d'<<24)+('o'<<16)+('u'<<8)+'b')          /* DOUB */
            //            && aff==SQLITE_AFF_NUMERIC ){
            //          aff = SQLITE_AFF_REAL;
            //#endif
            //        }else if( (h&0x00FFFFFF)==(('i'<<16)+('n'<<8)+'t') ){    /* INT */
            //          aff = SQLITE_AFF_INTEGER;
            //          break;
            //        }
            //      }

            //      return aff;
        }

        /*
        ** This routine is called by the parser while in the middle of
        ** parsing a CREATE TABLE statement.  The pFirst token is the first
        ** token in the sequence of tokens that describe the type of the
        ** column currently under construction.   pLast is the last token
        ** in the sequence.  Use this information to construct a string
        ** that contains the typename of the column and store that string
        ** in zType.
        */
        static void sqlite3AddColumnType(Parse pParse, Token pType)
        {
            Table p;
            Column pCol;

            p = pParse.pNewTable;
            if (p == null || NEVER(p.nCol < 1))
                return;
            pCol = p.aCol[p.nCol - 1];
            Debug.Assert(pCol.zType == null);
            pCol.zType = sqlite3NameFromToken(pParse.db, pType);
            pCol.affinity = sqlite3AffinityType(pCol.zType);
        }


        /*
        ** The expression is the default value for the most recently added column
        ** of the table currently under construction.
        **
        ** Default value expressions must be constant.  Raise an exception if this
        ** is not the case.
        **
        ** This routine is called by the parser while in the middle of
        ** parsing a CREATE TABLE statement.
        */
        static void sqlite3AddDefaultValue(Parse pParse, ExprSpan pSpan)
        {
            Table p;
            Column pCol;
            sqlite3 db = pParse.db;
            p = pParse.pNewTable;
            if (p != null)
            {
                pCol = (p.aCol[p.nCol - 1]);
                if (sqlite3ExprIsConstantOrFunction(pSpan.pExpr) == 0)
                {
                    sqlite3ErrorMsg(pParse, "default value of column [%s] is not constant",
                    pCol.zName);
                }
                else
                {
                    /* A copy of pExpr is used instead of the original, as pExpr contains
                    ** tokens that point to volatile memory. The 'span' of the expression
                    ** is required by pragma table_info.
                    */
                    sqlite3ExprDelete(db, ref pCol.pDflt);
                    pCol.pDflt = sqlite3ExprDup(db, pSpan.pExpr, EXPRDUP_REDUCE);
                    sqlite3DbFree(db, ref pCol.zDflt);
                    pCol.zDflt = pSpan.zStart.Substring(0, pSpan.zStart.Length - pSpan.zEnd.Length);
                    //sqlite3DbStrNDup( db, pSpan.zStart,
                    //                               (int)( pSpan.zEnd.Length - pSpan.zStart.Length ) );
                }
            }
            sqlite3ExprDelete(db, ref pSpan.pExpr);
        }

        /*
        ** Designate the PRIMARY KEY for the table.  pList is a list of names
        ** of columns that form the primary key.  If pList is NULL, then the
        ** most recently added column of the table is the primary key.
        **
        ** A table can have at most one primary key.  If the table already has
        ** a primary key (and this is the second primary key) then create an
        ** error.
        **
        ** If the PRIMARY KEY is on a single column whose datatype is INTEGER,
        ** then we will try to use that column as the rowid.  Set the Table.iPKey
        ** field of the table under construction to be the index of the
        ** INTEGER PRIMARY KEY column.  Table.iPKey is set to -1 if there is
        ** no INTEGER PRIMARY KEY.
        **
        ** If the key is not an INTEGER PRIMARY KEY, then create a unique
        ** index for the key.  No index is created for INTEGER PRIMARY KEYs.
        */
        // OVERLOADS, so I don't need to rewrite parse.c
        static void sqlite3AddPrimaryKey(Parse pParse, int null_2, int onError, int autoInc, int sortOrder)
        {
            sqlite3AddPrimaryKey(pParse, null, onError, autoInc, sortOrder);
        }
        static void sqlite3AddPrimaryKey(
        Parse pParse,    /* Parsing context */
        ExprList pList,  /* List of field names to be indexed */
        int onError,     /* What to do with a uniqueness conflict */
        int autoInc,    /* True if the AUTOINCREMENT keyword is present */
        int sortOrder   /* SQLITE_SO_ASC or SQLITE_SO_DESC */
        )
        {
            Table pTab = pParse.pNewTable;
            string zType = null;
            int iCol = -1, i;
            if (pTab == null || IN_DECLARE_VTAB(pParse))
                goto primary_key_exit;
            if ((pTab.tabFlags & TF_HasPrimaryKey) != 0)
            {
                sqlite3ErrorMsg(pParse,
                "table \"%s\" has more than one primary key", pTab.zName);
                goto primary_key_exit;
            }
            pTab.tabFlags |= TF_HasPrimaryKey;
            if (pList == null)
            {
                iCol = pTab.nCol - 1;
                pTab.aCol[iCol].isPrimKey = 1;
            }
            else
            {
                for (i = 0; i < pList.nExpr; i++)
                {
                    for (iCol = 0; iCol < pTab.nCol; iCol++)
                    {
                        if (pList.a[i].zName.Equals(pTab.aCol[iCol].zName, StringComparison.InvariantCultureIgnoreCase))
                        {
                            break;
                        }
                    }
                    if (iCol < pTab.nCol)
                    {
                        pTab.aCol[iCol].isPrimKey = 1;
                    }
                }
                if (pList.nExpr > 1)
                    iCol = -1;
            }
            if (iCol >= 0 && iCol < pTab.nCol)
            {
                zType = pTab.aCol[iCol].zType;
            }
            if (zType != null && zType.Equals("INTEGER", StringComparison.InvariantCultureIgnoreCase)
            && sortOrder == SQLITE_SO_ASC)
            {
                pTab.iPKey = iCol;
                pTab.keyConf = (byte)onError;
                Debug.Assert(autoInc == 0 || autoInc == 1);
                pTab.tabFlags |= (u8)(autoInc * TF_Autoincrement);
            }
            else if (autoInc != 0)
            {
#if !SQLITE_OMIT_AUTOINCREMENT
                sqlite3ErrorMsg(pParse, "AUTOINCREMENT is only allowed on an " +
                "INTEGER PRIMARY KEY");
#endif
            }
            else
            {
                Index p;
                p = sqlite3CreateIndex(pParse, 0, 0, 0, pList, onError, 0, 0, sortOrder, 0);
                if (p != null)
                {
                    p.autoIndex = 2;
                }
                pList = null;
            }

        primary_key_exit:
            sqlite3ExprListDelete(pParse.db, ref pList);
            return;
        }

        /*
        ** Add a new CHECK constraint to the table currently under construction.
        */
        static void sqlite3AddCheckConstraint(
        Parse pParse,    /* Parsing context */
        Expr pCheckExpr  /* The check expression */
        )
        {
            sqlite3 db = pParse.db;
#if !SQLITE_OMIT_CHECK
            Table pTab = pParse.pNewTable;
            if (pTab != null && !IN_DECLARE_VTAB(pParse))
            {
                pTab.pCheck = sqlite3ExprAnd(db, pTab.pCheck, pCheckExpr);
            }
            else
#endif
            {
                sqlite3ExprDelete(db, ref pCheckExpr);
            }
        }
        /*
        ** Set the collation function of the most recently parsed table column
        ** to the CollSeq given.
        */
        static void sqlite3AddCollateType(Parse pParse, Token pToken)
        {
            Table p;
            int i;
            string zColl;              /* Dequoted name of collation sequence */
            sqlite3 db;

            if ((p = pParse.pNewTable) == null)
                return;
            i = p.nCol - 1;
            db = pParse.db;
            zColl = sqlite3NameFromToken(db, pToken);
            if (zColl == null)
                return;

            if (sqlite3LocateCollSeq(pParse, zColl) != null)
            {
                Index pIdx;
                p.aCol[i].zColl = zColl;

                /* If the column is declared as "<name> PRIMARY KEY COLLATE <type>",
                ** then an index may have been created on this column before the
                ** collation type was added. Correct this if it is the case.
                */
                for (pIdx = p.pIndex; pIdx != null; pIdx = pIdx.pNext)
                {
                    Debug.Assert(pIdx.nColumn == 1);
                    if (pIdx.aiColumn[0] == i)
                    {
                        pIdx.azColl[0] = p.aCol[i].zColl;
                    }
                }
            }
            else
            {
                sqlite3DbFree(db, ref zColl);
            }
        }





        /*
        ** Generate a CREATE TABLE statement appropriate for the given
        ** table.  Memory to hold the text of the statement is obtained
        ** from sqliteMalloc() and must be freed by the calling function.
        */
        static string createTableStmt(sqlite3 db, Table p)
        {
            int i, k, n;
            StringBuilder zStmt;
            string zSep;
            string zSep2;
            string zEnd;
            Column pCol;
            n = 0;
            for (i = 0; i < p.nCol; i++)
            {//, pCol++){
                pCol = p.aCol[i];
                n += identLength(pCol.zName) + 5;
            }
            n += identLength(p.zName);
            if (n < 50)
            {
                zSep = "";
                zSep2 = ",";
                zEnd = ")";
            }
            else
            {
                zSep = "\n  ";
                zSep2 = ",\n  ";
                zEnd = "\n)";
            }
            n += 35 + 6 * p.nCol;
            zStmt = new StringBuilder(n);
            //zStmt = sqlite3DbMallocRaw(0, n);
            //if( zStmt==0 ){
            //  db.mallocFailed = 1;
            //  return 0;
            //}
            //sqlite3_snprintf(n, zStmt,"CREATE TABLE ");
            zStmt.Append("CREATE TABLE ");
            k = sqlite3Strlen30(zStmt);
            identPut(zStmt, ref k, p.zName);
            zStmt.Append('(');//zStmt[k++] = '(';
            for (i = 0; i < p.nCol; i++)
            {//, pCol++){
                pCol = p.aCol[i];
                string[] azType = new string[]  {
/* SQLITE_AFF_TEXT    */ " TEXT",
/* SQLITE_AFF_NONE    */ "",
/* SQLITE_AFF_NUMERIC */ " NUM",
/* SQLITE_AFF_INTEGER */ " INT",
/* SQLITE_AFF_REAL    */ " REAL"
};
                int len;
                string zType;

                zStmt.Append(zSep);//  sqlite3_snprintf(n-k, zStmt[k], zSep);
                k = sqlite3Strlen30(zStmt);//  k += strlen(zStmt[k]);
                zSep = zSep2;
                identPut(zStmt, ref k, pCol.zName);
                Debug.Assert(pCol.affinity - SQLITE_AFF_TEXT >= 0);
                Debug.Assert(pCol.affinity - SQLITE_AFF_TEXT < ArraySize(azType));
                testcase(pCol.affinity == SQLITE_AFF_TEXT);
                testcase(pCol.affinity == SQLITE_AFF_NONE);
                testcase(pCol.affinity == SQLITE_AFF_NUMERIC);
                testcase(pCol.affinity == SQLITE_AFF_INTEGER);
                testcase(pCol.affinity == SQLITE_AFF_REAL);

                zType = azType[pCol.affinity - SQLITE_AFF_TEXT];
                len = sqlite3Strlen30(zType);
                Debug.Assert(pCol.affinity == SQLITE_AFF_NONE
                || pCol.affinity == sqlite3AffinityType(zType));
                zStmt.Append(zType);// memcpy( &zStmt[k], zType, len );
                k += len;
                Debug.Assert(k <= n);
            }
            zStmt.Append(zEnd);//sqlite3_snprintf(n-k, zStmt[k], "%s", zEnd);
            return zStmt.ToString();
        }

        /*
        ** This routine is called to report the final ")" that terminates
        ** a CREATE TABLE statement.
        **
        ** The table structure that other action routines have been building
        ** is added to the internal hash tables, assuming no errors have
        ** occurred.
        **
        ** An entry for the table is made in the master table on disk, unless
        ** this is a temporary table or db.init.busy==1.  When db.init.busy==1
        ** it means we are reading the sqlite_master table because we just
        ** connected to the database or because the sqlite_master table has
        ** recently changed, so the entry for this table already exists in
        ** the sqlite_master table.  We do not want to create it again.
        **
        ** If the pSelect argument is not NULL, it means that this routine
        ** was called to create a table generated from a
        ** "CREATE TABLE ... AS SELECT ..." statement.  The column names of
        ** the new table will match the result set of the SELECT.
        */
        // OVERLOADS, so I don't need to rewrite parse.c
        static void sqlite3EndTable(Parse pParse, Token pCons, Token pEnd, int null_4)
        {
            sqlite3EndTable(pParse, pCons, pEnd, null);
        }
        static void sqlite3EndTable(Parse pParse, int null_2, int null_3, Select pSelect)
        {
            sqlite3EndTable(pParse, null, null, pSelect);
        }

        static void sqlite3EndTable(
        Parse pParse,          /* Parse context */
        Token pCons,           /* The ',' token after the last column defn. */
        Token pEnd,            /* The final ')' token in the CREATE TABLE */
        Select pSelect         /* Select from a "CREATE ... AS SELECT" */
        )
        {
            Table p;
            sqlite3 db = pParse.db;
            int iDb;

            if ((pEnd == null && pSelect == null) /*|| db.mallocFailed != 0 */ )
            {
                return;
            }
            p = pParse.pNewTable;
            if (p == null)
                return;

            Debug.Assert(0 == db.init.busy || pSelect == null);

            iDb = sqlite3SchemaToIndex(db, p.pSchema);

#if !SQLITE_OMIT_CHECK
            /* Resolve names in all CHECK constraint expressions.
*/
            if (p.pCheck != null)
            {
                SrcList sSrc;                   /* Fake SrcList for pParse.pNewTable */
                NameContext sNC;                /* Name context for pParse.pNewTable */

                sNC = new NameContext();// memset(sNC, 0, sizeof(sNC));
                sSrc = new SrcList();// memset(sSrc, 0, sizeof(sSrc));
                sSrc.nSrc = 1;
                sSrc.a = new SrcList_item[1];
                sSrc.a[0] = new SrcList_item();
                sSrc.a[0].zName = p.zName;
                sSrc.a[0].pTab = p;
                sSrc.a[0].iCursor = -1;
                sNC.pParse = pParse;
                sNC.pSrcList = sSrc;
                sNC.isCheck = 1;
                if (sqlite3ResolveExprNames(sNC, ref p.pCheck) != 0)
                {
                    return;
                }
            }
#endif // * !SQLITE_OMIT_CHECK) */

            /* If the db.init.busy is 1 it means we are reading the SQL off the
** "sqlite_master" or "sqlite_temp_master" table on the disk.
** So do not write to the disk again.  Extract the root page number
** for the table from the db.init.newTnum field.  (The page number
** should have been put there by the sqliteOpenCb routine.)
*/
            if (db.init.busy != 0)
            {
                p.tnum = db.init.newTnum;
            }

            /* If not initializing, then create a record for the new table
            ** in the SQLITE_MASTER table of the database.
            **
            ** If this is a TEMPORARY table, write the entry into the auxiliary
            ** file instead of into the main database file.
            */
            if (0 == db.init.busy)
            {
                int n;
                Vdbe v;
                String zType = "";    /* "view" or "table" */
                String zType2 = "";    /* "VIEW" or "TABLE" */
                String zStmt = "";    /* Text of the CREATE TABLE or CREATE VIEW statement */

                v = sqlite3GetVdbe(pParse);
                if (NEVER(v == null))
                    return;

                sqlite3VdbeAddOp1(v, OP_Close, 0);

                /*
                ** Initialize zType for the new view or table.
                */
                if (p.pSelect == null)
                {
                    /* A regular table */
                    zType = "table";
                    zType2 = "TABLE";
#if !SQLITE_OMIT_VIEW
                }
                else
                {
                    /* A view */
                    zType = "view";
                    zType2 = "VIEW";
#endif
                }

                /* If this is a CREATE TABLE xx AS SELECT ..., execute the SELECT
                ** statement to populate the new table. The root-page number for the
                ** new table is in register pParse->regRoot.
                **
                ** Once the SELECT has been coded by sqlite3Select(), it is in a
                ** suitable state to query for the column names and types to be used
                ** by the new table.
                **
                ** A shared-cache write-lock is not required to write to the new table,
                ** as a schema-lock must have already been obtained to create it. Since
                ** a schema-lock excludes all other database users, the write-lock would
                ** be redundant.
                */
                if (pSelect != null)
                {
                    SelectDest dest = new SelectDest();
                    Table pSelTab;

                    Debug.Assert(pParse.nTab == 1);
                    sqlite3VdbeAddOp3(v, OP_OpenWrite, 1, pParse.regRoot, iDb);
                    sqlite3VdbeChangeP5(v, 1);
                    pParse.nTab = 2;
                    sqlite3SelectDestInit(dest, SRT_Table, 1);
                    sqlite3Select(pParse, pSelect, ref dest);
                    sqlite3VdbeAddOp1(v, OP_Close, 1);
                    if (pParse.nErr == 0)
                    {
                        pSelTab = sqlite3ResultSetOfSelect(pParse, pSelect);
                        if (pSelTab == null)
                            return;
                        Debug.Assert(p.aCol == null);
                        p.nCol = pSelTab.nCol;
                        p.aCol = pSelTab.aCol;
                        pSelTab.nCol = 0;
                        pSelTab.aCol = null;
                        sqlite3DeleteTable(db, ref pSelTab);
                    }
                }

                /* Compute the complete text of the CREATE statement */
                if (pSelect != null)
                {
                    zStmt = createTableStmt(db, p);
                }
                else
                {
                    n = (int)(pParse.sNameToken.z.Length - pEnd.z.Length) + 1;
                    zStmt = sqlite3MPrintf(db,
                    "CREATE %s %.*s", zType2, n, pParse.sNameToken.z
                    );
                }

                /* A slot for the record has already been allocated in the
                ** SQLITE_MASTER table.  We just need to update that slot with all
                ** the information we've collected.
                */
                sqlite3NestedParse(pParse,
                "UPDATE %Q.%s " +
                "SET type='%s', name=%Q, tbl_name=%Q, rootpage=#%d, sql=%Q " +
                "WHERE rowid=#%d",
                db.aDb[iDb].zName, SCHEMA_TABLE(iDb),
                zType,
                p.zName,
                p.zName,
                pParse.regRoot,
                zStmt,
                pParse.regRowid
                );
                sqlite3DbFree(db, ref zStmt);
                sqlite3ChangeCookie(pParse, iDb);

#if !SQLITE_OMIT_AUTOINCREMENT
                /* Check to see if we need to create an sqlite_sequence table for
** keeping track of autoincrement keys.
*/
                if ((p.tabFlags & TF_Autoincrement) != 0)
                {
                    Db pDb = db.aDb[iDb];
                    Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
                    if (pDb.pSchema.pSeqTab == null)
                    {
                        sqlite3NestedParse(pParse,
                        "CREATE TABLE %Q.sqlite_sequence(name,seq)",
                        pDb.zName
                        );
                    }
                }
#endif

                /* Reparse everything to update our internal data structures */
                sqlite3VdbeAddParseSchemaOp(v, iDb,
                           sqlite3MPrintf(db, "tbl_name='%q'", p.zName));
            }


            /* Add the table to the in-memory representation of the database.
            */
            if (db.init.busy != 0)
            {
                Table pOld;
                Schema pSchema = p.pSchema;
                Debug.Assert(sqlite3SchemaMutexHeld(db, iDb, null));
                pOld = sqlite3HashInsert(ref pSchema.tblHash, p.zName,
                sqlite3Strlen30(p.zName), p);
                if (pOld != null)
                {
                    Debug.Assert(p == pOld);  /* Malloc must have failed inside HashInsert() */
                    //        db.mallocFailed = 1;
                    return;
                }
                pParse.pNewTable = null;
                db.nTable++;
                db.flags |= SQLITE_InternChanges;

#if !SQLITE_OMIT_ALTERTABLE
                if (p.pSelect == null)
                {
                    string zName = pParse.sNameToken.z;
                    int nName;
                    Debug.Assert(pSelect == null && pCons != null && pEnd != null);
                    if (pCons.z == null)
                    {
                        pCons = pEnd;
                    }
                    nName = zName.Length - pCons.z.Length;
                    p.addColOffset = 13 + nName; // sqlite3Utf8CharLen(zName, nName);
                }
#endif
            }
        }

        /*
        ** This routine is called to create a new foreign key on the table
        ** currently under construction.  pFromCol determines which columns
        ** in the current table point to the foreign key.  If pFromCol==0 then
        ** connect the key to the last column inserted.  pTo is the name of
        ** the table referred to.  pToCol is a list of tables in the other
        ** pTo table that the foreign key points to.  flags contains all
        ** information about the conflict resolution algorithms specified
        ** in the ON DELETE, ON UPDATE and ON INSERT clauses.
        **
        ** An FKey structure is created and added to the table currently
        ** under construction in the pParse.pNewTable field.
        **
        ** The foreign key is set for IMMEDIATE processing.  A subsequent call
        ** to sqlite3DeferForeignKey() might change this to DEFERRED.
        */
        // OVERLOADS, so I don't need to rewrite parse.c
        static void sqlite3CreateForeignKey(Parse pParse, int null_2, Token pTo, ExprList pToCol, int flags)
        {
            sqlite3CreateForeignKey(pParse, null, pTo, pToCol, flags);
        }
        static void sqlite3CreateForeignKey(
        Parse pParse,       /* Parsing context */
        ExprList pFromCol,  /* Columns in this table that point to other table */
        Token pTo,          /* Name of the other table */
        ExprList pToCol,    /* Columns in the other table */
        int flags           /* Conflict resolution algorithms. */
        )
        {
            sqlite3 db = pParse.db;
#if !SQLITE_OMIT_FOREIGN_KEY
            FKey pFKey = null;
            FKey pNextTo;
            Table p = pParse.pNewTable;
            int nByte;
            int i;
            int nCol;
            //string z;

            Debug.Assert(pTo != null);
            if (p == null || IN_DECLARE_VTAB(pParse))
                goto fk_end;
            if (pFromCol == null)
            {
                int iCol = p.nCol - 1;
                if (NEVER(iCol < 0))
                    goto fk_end;
                if (pToCol != null && pToCol.nExpr != 1)
                {
                    sqlite3ErrorMsg(pParse, "foreign key on %s" +
                    " should reference only one column of table %T",
                    p.aCol[iCol].zName, pTo);
                    goto fk_end;
                }
                nCol = 1;
            }
            else if (pToCol != null && pToCol.nExpr != pFromCol.nExpr)
            {
                sqlite3ErrorMsg(pParse,
                "number of columns in foreign key does not match the number of " +
                "columns in the referenced table");
                goto fk_end;
            }
            else
            {
                nCol = pFromCol.nExpr;
            }
            //nByte = sizeof(*pFKey) + (nCol-1)*sizeof(pFKey.aCol[0]) + pTo.n + 1;
            //if( pToCol ){
            //  for(i=0; i<pToCol.nExpr; i++){
            //    nByte += sqlite3Strlen30(pToCol->a[i].zName) + 1;
            //  }
            //}
            pFKey = new FKey();//sqlite3DbMallocZero(db, nByte );
            if (pFKey == null)
            {
                goto fk_end;
            }
            pFKey.pFrom = p;
            pFKey.pNextFrom = p.pFKey;
            //z = pFKey.aCol[nCol].zCol;
            pFKey.aCol = new FKey.sColMap[nCol];// z;
            pFKey.aCol[0] = new FKey.sColMap();
            pFKey.zTo = pTo.z.Substring(0, pTo.n);      //memcpy( z, pTo.z, pTo.n );
            //z[pTo.n] = 0;
            sqlite3Dequote(ref pFKey.zTo);
            //z += pTo.n + 1;
            pFKey.nCol = nCol;
            if (pFromCol == null)
            {
                pFKey.aCol[0].iFrom = p.nCol - 1;
            }
            else
            {
                for (i = 0; i < nCol; i++)
                {
                    if (pFKey.aCol[i] == null)
                        pFKey.aCol[i] = new FKey.sColMap();
                    int j;
                    for (j = 0; j < p.nCol; j++)
                    {
                        if (p.aCol[j].zName.Equals(pFromCol.a[i].zName, StringComparison.InvariantCultureIgnoreCase))
                        {
                            pFKey.aCol[i].iFrom = j;
                            break;
                        }
                    }
                    if (j >= p.nCol)
                    {
                        sqlite3ErrorMsg(pParse,
                        "unknown column \"%s\" in foreign key definition",
                        pFromCol.a[i].zName);
                        goto fk_end;
                    }
                }
            }
            if (pToCol != null)
            {
                for (i = 0; i < nCol; i++)
                {
                    int n = sqlite3Strlen30(pToCol.a[i].zName);
                    if (pFKey.aCol[i] == null)
                        pFKey.aCol[i] = new FKey.sColMap();
                    pFKey.aCol[i].zCol = pToCol.a[i].zName;
                    //memcpy( z, pToCol.a[i].zName, n );
                    //z[n] = 0;
                    //z += n + 1;
                }
            }
            pFKey.isDeferred = 0;
            pFKey.aAction[0] = (u8)(flags & 0xff);            /* ON DELETE action */
            pFKey.aAction[1] = (u8)((flags >> 8) & 0xff);    /* ON UPDATE action */

            Debug.Assert(sqlite3SchemaMutexHeld(db, 0, p.pSchema));
            pNextTo = sqlite3HashInsert(ref p.pSchema.fkeyHash,
                pFKey.zTo, sqlite3Strlen30(pFKey.zTo), pFKey
            );
            //if( pNextTo==pFKey ){
            //  db.mallocFailed = 1;
            //  goto fk_end;
            //}
            if (pNextTo != null)
            {
                Debug.Assert(pNextTo.pPrevTo == null);
                pFKey.pNextTo = pNextTo;
                pNextTo.pPrevTo = pFKey;
            }
            /* Link the foreign key to the table as the last step.
            */
            p.pFKey = pFKey;
            pFKey = null;

        fk_end:
            sqlite3DbFree(db, ref pFKey);
#endif // * !SQLITE_OMIT_FOREIGN_KEY) */
            sqlite3ExprListDelete(db, ref pFromCol);
            sqlite3ExprListDelete(db, ref pToCol);
        }






    }
}