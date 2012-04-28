#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
namespace FeaServer.Engine.Query
{
    public class Select
    {
        public enum SF : ushort
        {
            Distinct = 0x0001,  // Output should be DISTINCT
            Resolved = 0x0002,  // Identifiers have been resolved
            Aggregate = 0x0004,  // Contains aggregate functions
            UsesEphemeral = 0x0008,  // Uses the OpenEphemeral opcode
            Expanded = 0x0010,  // sqlite3SelectExpand() called on this
            HasTypeInfo = 0x0020,  // FROM subqueries have Table metadata
        }

        public enum SRT
        {
            Union = 1, // Store result as keys in an index
            Except = 2, // Remove result from a UNION index
            Exists = 3, // Store 1 if the result is not empty
            Discard = 4, // Do not save the results anywhere
            // The ORDER BY clause is ignored for all of the above
            Output = 5, // Output each row of result
            Mem = 6, // Store result in a memory cell
            Set = 7, // Store results as keys in an index
            Table = 8, // Store result as data with an automatic rowid
            EphemTab = 9, // Create transient tab and store like SRT_Table
            Coroutine = 10, // Generate a single row of result
        }

        public ExprList pEList;     // The fields of the result
        public Parser.TK op;        // One of: TK_UNION TK_ALL TK_INTERSECT TK_EXCEPT
        public char affinity;       // MakeRecord with this affinity for SRT_Set
        public SF selFlags;
        public SrcList pSrc;        // The FROM clause
        public Expr pWhere;         // The WHERE clause
        public ExprList pGroupBy;   // The GROUP BY clause
        public Expr pHaving;        // The HAVING clause
        public ExprList pOrderBy;   // The ORDER BY clause 
        public Select pPrior;       // Prior select in a compound select statement
        public Select pNext;        // Next select to the left in a compound
        public Select pRightmost;   // Right-most select in a compound select statement
        public Expr pLimit;         // LIMIT expression. NULL means not used.
        public Expr pOffset;        // OFFSET expression. NULL means not used.
        public int iLimit;
        public int iOffset;         // Memory registers holding LIMIT & OFFSET counters
        public int[] addrOpenEphm = new int[3]; // OP_OpenEphem opcodes related to this select
        public double nSelectRow;   // Estimated number of result rows

        public Select Copy()
        {
            if (this == null)
                return null;
            else
            {
                var cp = (Select)MemberwiseClone();
                if (pEList != null) cp.pEList = pEList.Copy();
                if (pSrc != null) cp.pSrc = pSrc.Copy();
                if (pWhere != null) cp.pWhere = pWhere.Copy();
                if (pGroupBy != null) cp.pGroupBy = pGroupBy.Copy();
                if (pHaving != null) cp.pHaving = pHaving.Copy();
                if (pOrderBy != null) cp.pOrderBy = pOrderBy.Copy();
                if (pPrior != null) cp.pPrior = pPrior.Copy();
                if (pNext != null) cp.pNext = pNext.Copy();
                if (pRightmost != null) cp.pRightmost = pRightmost.Copy();
                if (pLimit != null) cp.pLimit = pLimit.Copy();
                if (pOffset != null) cp.pOffset = pOffset.Copy();
                return cp;
            }
        }

        //public static bool IgnorableOrderby(object x)
        //{
        //    //#define IgnorableOrderby(X) ((X->eDest)<=SRT_Discard)
        //    return false;
        //}
    }
}
