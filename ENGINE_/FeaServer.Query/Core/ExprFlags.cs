using System;
namespace FeaServer.Core
{
    [Flags]
    public enum ExprFlags : ushort
    {
        EP_FromJoin = 0x0001,   /* Originated in ON or USING clause of a join */
        EP_Agg = 0x0002,        /* Contains one or more aggregate functions */
        EP_Resolved = 0x0004,   /* IDs have been resolved to COLUMNs */
        EP_Error = 0x0008,      /* Expression contains one or more errors */
        EP_Distinct = 0x0010,   /* Aggregate function with DISTINCT keyword */
        EP_VarSelect = 0x0020,  /* pSelect is correlated, not constant */
        EP_DblQuoted = 0x0040,  /* token.z was originally in "..." */
        EP_InfixFunc = 0x0080,  /* True for an infix function: LIKE, GLOB, etc */
        EP_ExpCollate = 0x0100, /* Collating sequence specified explicitly */
        EP_FixedDest = 0x0200,  /* Result needed in a specific register */
        EP_IntValue = 0x0400,   /* Integer value contained in u.iValue */
        EP_xIsSelect = 0x0800,  /* x.pSelect is valid (otherwise x.pList is) */
        EP_Hint = 0x1000,       /* Optimizer hint. Not required for correctness */
        EP_Reduced = 0x2000,    /* Expr struct is EXPR_REDUCEDSIZE bytes only */
        EP_TokenOnly = 0x4000,  /* Expr struct is EXPR_TOKENONLYSIZE bytes only */
        EP_Static = 0x8000,     /* Held in memory not obtained from malloc() */
    }
}
