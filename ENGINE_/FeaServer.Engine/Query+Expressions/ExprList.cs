namespace FeaServer.Engine.Query
{
    // A list of expressions.  Each expression may optionally have a name.  An expr/name combination can be used in several ways, such
    // as the list of "expr AS ID" fields following a "SELECT" or in the list of "ID = expr" items in an UPDATE.  A list of expressions can
    // also be used as the argument to a function, in which case the a.zName field is not used.
    public class ExprList
    {
        public class item
        {
            public Expr pExpr;          /* The list of expressions */
            public string zName;        /* Token associated with this expression */
            public string zSpan;        /*  Original text of the expression */
            public byte sortOrder;        /* 1 for DESC or 0 for ASC */
            public byte done;             /* A flag to indicate when processing is finished */
            public ushort iCol;            /* For ORDER BY, column number in result set */
            public ushort iAlias;          /* Index into Parse.aAlias[] for zName */
        }

        public int nExpr;             /* Number of expressions on the list */
        public int nAlloc;            /* Number of entries allocated below */
        public int iECursor;          /* VDBE VdbeCursor associated with this ExprList */
        public item[] a;     /* One entry for each expression */

        public ExprList Copy()
        {
            if (this == null)
                return null;
            else
            {
                var cp = (ExprList)MemberwiseClone();
                a.CopyTo(cp.a, 0);
                return cp;
            }
        }
    }
}