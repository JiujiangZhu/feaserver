namespace FeaServer.Core
{
    // An instance of this structure is used by the parser to record both the parse tree for an expression and the span of input text for an expression.
    public struct ExprSpan
    {
        public Expr pExpr;  /* The expression parse tree */
        public int zStart;  /* First character of input text */
        public int zEnd;    /* One character past the end of input text */
    }
}


