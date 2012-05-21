using FeaServer.Core;
namespace FeaServer
{
    public class Select
    {
        public ExprList pEList;      /* The fields of the result */
        public byte op;                 /* One of: TK_UNION TK_ALL TK_INTERSECT TK_EXCEPT */
        public char affinity;         /* MakeRecord with this affinity for SRT_Set */
        public ushort selFlags;          /* Various SF_* values */
        public int iLimit, iOffset;   /* Memory registers holding LIMIT & OFFSET counters */
        public int[] addrOpenEphm = new int[3];   /* OP_OpenEphem opcodes related to this select */
        public double nSelectRow;     /* Estimated number of result rows */
        public SrcList pSrc;         /* The FROM clause */
        public Expr pWhere;          /* The WHERE clause */
        public ExprList pGroupBy;    /* The GROUP BY clause */
        public Expr pHaving;         /* The HAVING clause */
        public ExprList pOrderBy;    /* The ORDER BY clause */
        public Select pPrior;        /* Prior select in a compound select statement */
        public Select pNext;         /* Next select to the left in a compound */
        public Select pRightmost;    /* Right-most select in a compound select statement */
        public Expr pLimit;          /* LIMIT expression. NULL means not used. */
        public Expr pOffset;         /* OFFSET expression. NULL means not used. */
    }
}
