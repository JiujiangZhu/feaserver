using FeaServer.Query;
using System.Runtime.InteropServices;
namespace FeaServer.Core
{
    /*
    ** Each node of an expression in the parse tree is an instance of this structure.
    **
    ** Expr.op is the opcode. The integer parser token codes are reused as opcodes here. For example, the parser defines TK_GE to be an integer
    ** code representing the ">=" operator. This same integer code is reused to represent the greater-than-or-equal-to operator in the expression
    ** tree.
    **
    ** If the expression is an SQL literal (TK_INTEGER, TK_FLOAT, TK_BLOB, or TK_STRING), then Expr.token contains the text of the SQL literal. If
    ** the expression is a variable (TK_VARIABLE), then Expr.token contains the variable name. Finally, if the expression is an SQL function (TK_FUNCTION),
    ** then Expr.token contains the name of the function.
    **
    ** Expr.pRight and Expr.pLeft are the left and right subexpressions of a binary operator. Either or both may be NULL.
    **
    ** Expr.x.pList is a list of arguments if the expression is an SQL function, a CASE expression or an IN expression of the form "<lhs> IN (<y>, <z>...)".
    ** Expr.x.pSelect is used if the expression is a sub-select or an expression of the form "<lhs> IN (SELECT ...)". If the EP_xIsSelect bit is set in the
    ** Expr.flags mask, then Expr.x.pSelect is valid. Otherwise, Expr.x.pList is valid.
    **
    ** An expression of the form ID or ID.ID refers to a column in a table. For such expressions, Expr.op is set to TK_COLUMN and Expr.iTable is
    ** the integer cursor number of a VDBE cursor pointing to that table and Expr.iColumn is the column number for the specific column.  If the
    ** expression is used as a result in an aggregate SELECT, then the value is also stored in the Expr.iAgg column in the aggregate so that
    ** it can be accessed after all aggregates are computed.
    **
    ** If the expression is an unbound variable marker (a question mark character '?' in the original SQL) then the Expr.iTable holds the index 
    ** number for that variable.
    **
    ** If the expression is a subquery then Expr.iColumn holds an integer register number containing the result of the subquery.  If the
    ** subquery gives a constant result, then iTable is -1.  If the subquery gives a different answer at different times during statement processing
    ** then iTable is the address of a subroutine that computes the subquery.
    **
    ** If the Expr is of type OP_Column, and the table it is selecting from is a disk table or the "old.*" pseudo-table, then pTab points to the
    ** corresponding table definition.
    **
    ** ALLOCATION NOTES:
    **
    ** Expr objects can use a lot of memory space in database schema.  To help reduce memory requirements, sometimes an Expr object will be
    ** truncated.  And to reduce the number of memory allocations, sometimes two or more Expr objects will be stored in a single memory allocation,
    ** together with Expr.zToken strings.
    **
    ** If the EP_Reduced and EP_TokenOnly flags are set when an Expr object is truncated.  When EP_Reduced is set, then all
    ** the child Expr objects in the Expr.pLeft and Expr.pRight subtrees are contained within the same memory allocation.  Note, however, that
    ** the subtrees in Expr.x.pList or Expr.x.pSelect are always separately allocated, regardless of whether or not EP_Reduced is set.
    */
    public class Expr
    {
        [StructLayout(LayoutKind.Explicit)]
        public struct U
        {
            [FieldOffset(0)]
            public string Token;    // Token value. Zero terminated and dequoted
            [FieldOffset(0)]
            public int Value;       // Non-negative integer value if EP_IntValue
        }
        public byte op;             // Operation performed by this node
        public char affinity;       // The affinity of the column or 0 if not a column
        public ExprFlags flags;     // Various flags.  EP_* See below
        public U u;
    }

    // If the EP_TokenOnly flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to access them will result in a segfault or malfunction. 
    public class Expr2 : Expr
    {
        [StructLayout(LayoutKind.Explicit)]
        public struct X
        {
            [FieldOffset(0)]
            public ExprList pList;      // Function arguments or in "<expr> IN (<expr-list)"
            [FieldOffset(0)]
            public Select pSelect;      // Used for sub-selects and "<expr> IN (<select>)"
        }
        public Expr Left;           // Left subnode
        public Expr Right;          // Right subnode
        public X x;
        public CollSeq pColl;       // The collation type of the column or 0
    }

    // If the EP_Reduced flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to access them will result in a segfault or malfunction.
    public class Expr3 : Expr2
    {
        public int iTable;          // TK_COLUMN: cursor number of table holding column | TK_REGISTER: register number | TK_TRIGGER: 1 -> new, 0 -> old
        public short iColumn;       // TK_COLUMN: column index.  -1 for rowid. | TK_VARIABLE: variable number (always >= 1).
        public short iAgg;          // Which entry in pAggInfo->aCol[] or ->aFunc[]
        public short iRightJoinTable; // If EP_FromJoin, the right table of the joi
        public ExprFlags2 flags2;   // Second set of flags.  EP2_...
        public byte op2;            // If a TK_REGISTER, the original value of Expr.op
        public AggInfo pAggInfo;    // Used by TK_AGG_COLUMN and TK_AGG_FUNCTION
        public Table pTab;          // Table for TK_COLUMN expressions.
        public int nHeight;         // Height of the tree headed by this node
    }
}
