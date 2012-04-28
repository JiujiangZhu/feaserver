namespace FeaServer.Engine.Query
{
    // An instance of this structure is used by the parser to record both the parse tree for an expression and the span of input text for an expression.
    public class ExprSpan
    {
        public Expr pExpr;      // The expression parse tree
        public string zStart;   // First character of input text
        public string zEnd;     // One character past the end of input text
    }
}