using System;
using System.Diagnostics;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // Expand the space allocated for the given SrcList object by creating nExtra new slots beginning at iStart.  iStart is zero based.
        // New slots are zeroed.
        // For example, suppose a SrcList initially contains two entries: A,B. To append 3 new entries onto the end, do this:
        //    sqlite3SrcListEnlarge(db, pSrclist, 3, 2);
        // After the call above it would contain:  A, B, nil, nil, nil. If the iStart argument had been 1 instead of 2, then the result
        // would have been:  A, nil, nil, nil, B.  To prepend the new slots, the iStart value would be 0.  The result then would
        // be: nil, nil, nil, A, B.
        // If a memory allocation fails the SrcList is unchanged.  The db.mallocFailed flag will be set to true.
        internal static SrcList sqlite3SrcListEnlarge(sqlite3 db, SrcList pSrc, int nExtra, int iStart)
        {
            Debug.Assert(iStart >= 0);
            Debug.Assert(nExtra >= 1);
            Debug.Assert(pSrc != null);
            Debug.Assert(iStart <= pSrc.nSrc);
            // Allocate additional space if needed
            if (pSrc.nSrc + nExtra > pSrc.nAlloc)
            {
                var nAlloc = pSrc.nSrc + nExtra;
                pSrc.nAlloc = (short)nAlloc;
                Array.Resize(ref pSrc.a, nAlloc);
            }
            // Move existing slots that come after the newly inserted slots out of the way
            for (var i = pSrc.nSrc - 1; i >= iStart; i--)
                pSrc.a[i + nExtra] = pSrc.a[i];
            pSrc.nSrc += (short)nExtra;
            // Zero the newly allocated slots
            for (var i = iStart; i < iStart + nExtra; i++)
            {
                pSrc.a[i] = new SrcList_item();
                pSrc.a[i].iCursor = -1;
            }
            return pSrc;
        }


        // Append a new table name to the given SrcList.  Create a new SrcList if need be.  A new entry is created in the SrcList even if pTable is NULL.
        //
        // A SrcList is returned, or NULL if there is an OOM error.  The returned SrcList might be the same as the SrcList that was input or it might be
        // a new one.  If an OOM error does occurs, then the prior value of pList that is input to this routine is automatically freed.
        //
        // If pDatabase is not null, it means that the table has an optional database name prefix.  Like this:  "database.table".  The pDatabase
        // points to the table name and the pTable points to the database name. The SrcList.a[].zName field is filled with the table name which might
        // come from pTable (if pDatabase is NULL) or from pDatabase. SrcList.a[].zDatabase is filled with the database name from pTable,
        // or with NULL if no database is specified.
        // In other words, if call like this:
        //         sqlite3SrcListAppend(D,A,B,0);
        // Then B is a table name and the database name is unspecified.  If called like this:
        //         sqlite3SrcListAppend(D,A,B,C);
        // Then C is the table name and B is the database name.  If C is defined then so is B.  In other words, we never have a case where:
        //         sqlite3SrcListAppend(D,A,0,C);
        // Both pTable and pDatabase are assumed to be quoted.  They are dequoted before being added to the SrcList.
        internal static SrcList sqlite3SrcListAppend(sqlite3 db, int null_2, Token pTable, int null_4) { return sqlite3SrcListAppend(db, null, pTable, null); }
        internal static SrcList sqlite3SrcListAppend(sqlite3 db, int null_2, Token pTable, Token pDatabase) { return sqlite3SrcListAppend(db, null, pTable, pDatabase); }
        internal static SrcList sqlite3SrcListAppend(sqlite3 db, SrcList pList, Token pTable, Token pDatabase)
        {
            Debug.Assert(pDatabase == null || pTable != null);  // Cannot have C without B
            if (pList == null)
            {
                pList = new SrcList();
                pList.nAlloc = 1;
                pList.a = new SrcList_item[1];
            }
            pList = sqlite3SrcListEnlarge(db, pList, 1, pList.nSrc);
            var pItem = pList.a[pList.nSrc - 1];
            if (pDatabase != null && String.IsNullOrEmpty(pDatabase.z))
                pDatabase = null;
            if (pDatabase != null)
            {
                var pTemp = pDatabase;
                pDatabase = pTable;
                pTable = pTemp;
            }
            pItem.zName = sqlite3NameFromToken(db, pTable);
            pItem.zDatabase = sqlite3NameFromToken(db, pDatabase);
            return pList;
        }

        // Assign VdbeCursor index numbers to all tables in a SrcList
        internal static void sqlite3SrcListAssignCursors(Parse pParse, SrcList pList)
        {
            Debug.Assert(pList != null);
            if (pList != null)
                for (var i = 0; i < pList.nSrc; i++)
                {
                    var pItem = pList.a[i];
                    if (pItem.iCursor >= 0)
                        break;
                    pItem.iCursor = pParse.nTab++;
                    if (pItem.pSelect != null)
                        sqlite3SrcListAssignCursors(pParse, pItem.pSelect.pSrc);
                }
        }

        // Delete an entire SrcList including all its substructure.
        internal static void sqlite3SrcListDelete(sqlite3 db, ref SrcList pList)
        {
            if (pList == null)
                return;
            for (var i = 0; i < pList.nSrc; i++)
            {
                var pItem = pList.a[i];
                sqlite3DbFree(db, ref pItem.zDatabase);
                sqlite3DbFree(db, ref pItem.zName);
                sqlite3DbFree(db, ref pItem.zAlias);
                sqlite3DbFree(db, ref pItem.zIndex);
                sqlite3DeleteTable(db, ref pItem.pTab);
                sqlite3SelectDelete(db, ref pItem.pSelect);
                sqlite3ExprDelete(db, ref pItem.pOn);
                sqlite3IdListDelete(db, ref pItem.pUsing);
            }
            sqlite3DbFree(db, ref pList);
        }

        // This routine is called by the parser to add a new term to the end of a growing FROM clause.  The "p" parameter is the part of
        // the FROM clause that has already been constructed.  "p" is NULL if this is the first term of the FROM clause.  pTable and pDatabase
        // are the name of the table and database named in the FROM clause term. pDatabase is NULL if the database name qualifier is missing - the
        // usual case.  If the term has a alias, then pAlias points to the alias token.  If the term is a subquery, then pSubquery is the
        // SELECT statement that the subquery encodes.  The pTable and pDatabase parameters are NULL for subqueries.  The pOn and pUsing
        // parameters are the content of the ON and USING clauses.
        //
        // Return a new SrcList which encodes is the FROM with the new term added.
        internal static SrcList sqlite3SrcListAppendFromTerm(Parse pParse, SrcList p, int null_3, int null_4, Token pAlias, Select pSubquery, Expr pOn, IdList pUsing) { return sqlite3SrcListAppendFromTerm(pParse, p, null, null, pAlias, pSubquery, pOn, pUsing); }
        internal static SrcList sqlite3SrcListAppendFromTerm(Parse pParse, SrcList p, Token pTable, Token pDatabase, Token pAlias, int null_6, Expr pOn, IdList pUsing) { return sqlite3SrcListAppendFromTerm(pParse, p, pTable, pDatabase, pAlias, null, pOn, pUsing); }
        internal static SrcList sqlite3SrcListAppendFromTerm(Parse pParse, SrcList p, Token pTable, Token pDatabase, Token pAlias, Select pSubquery, Expr pOn, IdList pUsing)
        {
            var db = pParse.db;
            if (null == p && (pOn != null || pUsing != null))
            {
                sqlite3ErrorMsg(pParse, "a JOIN clause is required before %s", pOn != null ? "ON" : "USING");
                goto append_from_error;
            }
            p = sqlite3SrcListAppend(db, p, pTable, pDatabase);
            var pItem = p.a[p.nSrc - 1];
            Debug.Assert(pAlias != null);
            if (pAlias.n != 0)
                pItem.zAlias = sqlite3NameFromToken(db, pAlias);
            pItem.pSelect = pSubquery;
            pItem.pOn = pOn;
            pItem.pUsing = pUsing;
            return p;
        append_from_error:
            Debug.Assert(p == null);
            sqlite3ExprDelete(db, ref pOn);
            sqlite3IdListDelete(db, ref pUsing);
            sqlite3SelectDelete(db, ref pSubquery);
            return null;
        }

        // Add an INDEXED BY or NOT INDEXED clause to the most recently added element of the source-list passed as the second argument.
        internal static void sqlite3SrcListIndexedBy(Parse pParse, SrcList p, Token pIndexedBy)
        {
            Debug.Assert(pIndexedBy != null);
            if (p != null && Check.ALWAYS(p.nSrc > 0))
            {
                var pItem = p.a[p.nSrc - 1];
                Debug.Assert(0 == pItem.notIndexed && pItem.zIndex == null);
                if (pIndexedBy.n == 1 && null == pIndexedBy.z)
                    // A "NOT INDEXED" clause was supplied. See parse.y construct "indexed_opt" for details.
                    pItem.notIndexed = 1;
                else
                    pItem.zIndex = sqlite3NameFromToken(pParse.db, pIndexedBy);
            }
        }

        // When building up a FROM clause in the parser, the join operator is initially attached to the left operand.  But the code generator
        // expects the join operator to be on the right operand.  This routine Shifts all join operators from left to right for an entire FROM
        // clause.
        // Example: Suppose the join is like this:
        //           A natural cross join B
        // The operator is "natural cross join".  The A and B operands are stored in p.a[0] and p.a[1], respectively.  The parser initially stores the
        // operator with A.  This routine shifts that operator over to B.
        internal static void sqlite3SrcListShiftJoinType(SrcList p)
        {
            if (p != null && p.a != null)
            {
                for (var i = p.nSrc - 1; i > 0; i--)
                    p.a[i].jointype = p.a[i - 1].jointype;
                p.a[0].jointype = 0;
            }
        }
    }
}