#define SQLITE_MAX_EXPR_DEPTH
#if !SQLITE_MAX_VARIABLE_NUMBER
using ynVar = System.Int16;
using System;
using FeaServer.Engine.Core;
#else
using ynVar = System.Int32; 
#endif
namespace FeaServer.Engine.Query
{
    // Each node of an expression in the parse tree is an instance of this structure.
    //
    // Expr.op is the opcode.  The integer parser token codes are reused as opcodes here.  For example, the parser defines TK_GE to be an integer
    // code representing the ">=" operator.  This same integer code is reused to represent the greater-than-or-equal-to operator in the expression
    // tree.
    //
    // If the expression is an SQL literal (TK_INTEGER, TK_FLOAT, TK_BLOB, or TK_STRING), then Expr.token contains the text of the SQL literal. If
    // the expression is a variable (TK_VARIABLE), then Expr.token contains the variable name. Finally, if the expression is an SQL function (TK_FUNCTION),
    // then Expr.token contains the name of the function.
    //
    // Expr.pRight and Expr.pLeft are the left and right subexpressions of a binary operator. Either or both may be NULL.
    //
    // Expr.x.pList is a list of arguments if the expression is an SQL function, a CASE expression or an IN expression of the form "<lhs> IN (<y>, <z>...)".
    // Expr.x.pSelect is used if the expression is a sub-select or an expression of the form "<lhs> IN (SELECT ...)". If the EP_xIsSelect bit is set in the
    // Expr.flags mask, then Expr.x.pSelect is valid. Otherwise, Expr.x.pList is valid.
    //
    // An expression of the form ID or ID.ID refers to a column in a table. For such expressions, Expr.op is set to TK_COLUMN and Expr.iTable is
    // the integer cursor number of a VDBE cursor pointing to that table and Expr.iColumn is the column number for the specific column.  If the
    // expression is used as a result in an aggregate SELECT, then the value is also stored in the Expr.iAgg column in the aggregate so that
    // it can be accessed after all aggregates are computed.
    //
    // If the expression is an unbound variable marker (a question mark character '?' in the original SQL) then the Expr.iTable holds the index
    // number for that variable.
    //
    // If the expression is a subquery then Expr.iColumn holds an integer register number containing the result of the subquery.  If the
    // subquery gives a constant result, then iTable is -1.  If the subquery gives a different answer at different times during statement processing
    // then iTable is the address of a subroutine that computes the subquery.
    //
    // If the Expr is of type OP_Column, and the table it is selecting from is a disk table or the "old.*" pseudo-table, then pTab points to the
    // corresponding table definition.
    //
    // ALLOCATION NOTES:
    //
    // Expr objects can use a lot of memory space in database schema.  To help reduce memory requirements, sometimes an Expr object will be
    // truncated.  And to reduce the number of memory allocations, sometimes two or more Expr objects will be stored in a single memory allocation,
    // together with Expr.zToken strings.
    //
    // If the EP_Reduced and EP_TokenOnly flags are set when an Expr object is truncated.  When EP_Reduced is set, then all
    // the child Expr objects in the Expr.pLeft and Expr.pRight subtrees are contained within the same memory allocation.  Note, however, that
    // the subtrees in Expr.x.pList or Expr.x.pSelect are always separately allocated, regardless of whether or not EP_Reduced is set.
    public partial class Expr
    {
        [Flags]
        public enum EP : ushort
        {
            // The following are the meanings of bits in the Expr.flags field.
            FromJoin = 0x0001,  // Originated in ON or USING clause of a join
            Agg = 0x0002,       // Contains one or more aggregate functions
            Resolved = 0x0004,  // IDs have been resolved to COLUMNs
            Error = 0x0008,     // Expression contains one or more errors
            Distinct = 0x0010,  // Aggregate function with DISTINCT keyword
            VarSelect = 0x0020, // pSelect is correlated, not constant
            DblQuoted = 0x0040, // token.z was originally in "..."
            InfixFunc = 0x0080, // True for an infix function: LIKE, GLOB, etc
            ExpCollate = 0x0100,// Collating sequence specified explicitly
            FixedDest = 0x0200, // Result needed in a specific register
            IntValue = 0x0400,  // Integer value contained in u.iValue
            xIsSelect = 0x0800, // x.pSelect is valid (otherwise x.pList is)
            //
            Reduced = 0x1000,   // Expr struct is EXPR_REDUCEDSIZE bytes only
            TokenOnly = 0x2000, // Expr struct is EXPR_TOKENONLYSIZE bytes only
            Static = 0x4000,    // Held in memory not obtained from malloc()
        }

        // The following are the meanings of bits in the Expr.flags2 field.
        internal const byte EP2_MallocedToken = 0x0001;  // Need to sqlite3DbFree() Expr.zToken
        internal const byte EP2_Irreducible = 0x0002;    // Cannot EXPRDUP_REDUCE this Expr

        // The pseudo-routine sqlite3ExprSetIrreducible sets the EP2_Irreducible flag on an expression structure.  This flag is used for VV&A only.  The
        // routine is implemented as a macro that only works when in debugging mode, so as not to burden production code.
#if DEBUG
        internal static void ExprSetIrreducible(Expr X) { X.flags2 |= EP2_Irreducible; }
#else
        internal static void ExprSetIrreducible(Expr X) { }
#endif

        // These macros can be used to test, set, or clear bits in the Expr.flags field.
        internal static bool ExprHasProperty(Expr E, EP P) { return (E.flags & P) == P; }
        internal static bool ExprHasAnyProperty(Expr E, EP P) { return (E.flags & P) != 0; }
        internal static void ExprSetProperty(Expr E, EP P) { E.flags = (E.flags | P); }
        internal static void ExprClearProperty(Expr E, EP P) { E.flags = (E.flags & ~P); }

        // Macros to determine the number of bytes required by a normal Expr struct, an Expr struct with the EP_Reduced flag set in Expr.flags
        // and an Expr struct with the EP_TokenOnly flag set.
        // We don't use these in C#, but define them anyway,
        internal const int EXPR_FULLSIZE = 48;       // sizeof(Expr)           // Full size
        internal const int EXPR_REDUCEDSIZE = 24;    // offsetof(Expr,iTable)  // Common features
        internal const int EXPR_TOKENONLYSIZE = 8;  // offsetof(Expr,pLeft)   // Fewer features

        // Flags passed to the sqlite3ExprDup() function. See the header comment above sqlite3ExprDup() for details.
        internal const int EXPRDUP_REDUCE = 0x0001; // Used reduced-size Expr nodes

        #region EP_TokenOnly
        public Parser.TK op;        // Operation performed by this node
        public char affinity;       // The affinity of the column or 0 if not a column
        public EP flags;     
        public struct _u
        {
            public string zToken;   // Token value. Zero terminated and dequoted
            public int iValue;      // Non-negative integer value if EP_IntValue
        }
        public _u u;
        #endregion
        // If the EP_TokenOnly flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to access them will result in a segfault or malfunction.
        #region EP_Reduced
        public Expr pLeft;          // Left subnode
        public Expr pRight;         // Right subnode
        public struct _x
        {
            public ExprList pList;  // Function arguments or in "<expr> IN (<expr-list)"
            public Select pSelect;  // Used for sub-selects and "<expr> IN (<select>)"
        }
        public _x x;
        public CollSeq pColl;       // The collation type of the column or 0
        #endregion
        // If the EP_Reduced flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to access them will result in a segfault or malfunction.
        #region Remaining
        public int iTable;            // TK_COLUMN: cursor number of table holding column | TK_REGISTER: register number | TK_TRIGGER: 1 -> new, 0 -> old
        public ynVar iColumn;         // TK_COLUMN: column index.  -1 for rowid. | TK_VARIABLE: variable number (always >= 1).
        public short iAgg;              // Which entry in pAggInfo->aCol[] or ->aFunc[]
        public short iRightJoinTable;   // If EP_FromJoin, the right table of the join
        public byte flags2;             // Second set of flags.  EP2_...
        public byte op2;                // If a TK_REGISTER, the original value of Expr.op
        public AggInfo pAggInfo;      // Used by TK_AGG_COLUMN and TK_AGG_FUNCTION
        public ITable pTab;            // Table for TK_COLUMN expressions.
#if SQLITE_MAX_EXPR_DEPTH
        public int nHeight;           // Height of the tree headed by this node
        public ITable pZombieTab;      // List of Table objects to delete after code gen
#endif
        #endregion

        public void CopyFrom(Expr cf)
        {
            op = cf.op;
            affinity = cf.affinity;
            flags = cf.flags;
            u = cf.u;
            pColl = (cf.pColl == null ? null : cf.pColl.Copy());
            iTable = cf.iTable;
            iColumn = cf.iColumn;
            pAggInfo = (cf.pAggInfo == null ? null : cf.pAggInfo.Copy());
            iAgg = cf.iAgg;
            iRightJoinTable = cf.iRightJoinTable;
            flags2 = cf.flags2;
            pTab = (cf.pTab == null ? null : cf.pTab);
#if SQLITE_MAX_EXPR_DEPTH
            nHeight = cf.nHeight;
            pZombieTab = cf.pZombieTab;
#endif
            pLeft = (cf.pLeft == null ? null : cf.pLeft.Copy());
            pRight = (cf.pRight == null ? null : cf.pRight.Copy());
            x.pList = (cf.x.pList == null ? null : cf.x.pList.Copy());
            x.pSelect = (cf.x.pSelect == null ? null : cf.x.pSelect.Copy());
        }

        public Expr Copy() { return (this == null ? null : Copy(flags)); }
        public Expr Copy(EP flag)
        {
            #region EP_TokenOnly
            var cp = new Expr
            {
                op = op,
                affinity = affinity,
                flags = flags,
                u = u,
            };
            if ((flag & EP.TokenOnly) != 0) return cp;
            #endregion
            #region EP_Reduced
            if (pLeft != null)
                cp.pLeft = pLeft.Copy();
            if (pRight != null)
                cp.pRight = pRight.Copy();
            cp.x = x;
            cp.pColl = pColl;
            if ((flag & EP.Reduced) != 0) return cp;
            #endregion
            #region Remaining
            cp.iTable = iTable;
            cp.iColumn = iColumn;
            cp.iAgg = iAgg;
            cp.iRightJoinTable = iRightJoinTable;
            cp.flags2 = flags2;
            cp.op2 = op2;
            cp.pAggInfo = pAggInfo;
            cp.pTab = pTab;
#if SQLITE_MAX_EXPR_DEPTH
            cp.nHeight = nHeight;
            cp.pZombieTab = pZombieTab;
#endif
            return cp;
            #endregion
        }
    }
}