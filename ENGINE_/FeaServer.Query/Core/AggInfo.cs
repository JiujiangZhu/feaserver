using System;
namespace FeaServer.Core
{
    /*
    ** An instance of this structure contains information needed to generate code for a SELECT that contains aggregate functions.
    **
    ** If Expr.op==TK_AGG_COLUMN or TK_AGG_FUNCTION then Expr.pAggInfo is a pointer to this structure.  The Expr.iColumn field is the index in
    ** AggInfo.aCol[] or AggInfo.aFunc[] of information needed to generate code for that node.
    **
    ** AggInfo.pGroupBy and AggInfo.aFunc.pExpr point to fields within the original Select structure that describes the SELECT statement.  These
    ** fields do not need to be freed when deallocating the AggInfo structure.
    */
    public class AggInfo
    {
        // For each column used in source tables
        public struct Col
        {
            public Table pTab;             /* Source table */
            public int iTable;             /* Cursor number of the source table */
            public int iColumn;            /* Column number within the source table */
            public int iSorterColumn;      /* Column number in the sorting index */
            public int iMem;               /* Memory location that acts as accumulator */
            public Expr pExpr;             /* The original expression */
        }
        // For each aggregate function
        public struct Func
        {
            public Expr pExpr;             /* Expression encoding the function */
            public FuncDef pFunc;          /* The aggregate function implementation */
            public int iMem;               /* Memory location that acts as accumulator */
            public int iDistinct;          /* Ephemeral table used to enforce DISTINCT */
        }
        public byte directMode;     /* Direct rendering mode means take data directly from source tables rather than from accumulators */
        public byte useSortingIdx;  /* In direct mode, reference the sorting index rather than the source table */
        public int sortingIdx;      /* Cursor number of the sorting index */
        public int sortingIdxPTab;  /* Cursor number of pseudo-table */
        public int nSortingColumn;  /* Number of columns in the sorting index */
        public ExprList pGroupBy;   /* The group by clause */
        public Col aCol;
        public int nColumn;         /* Number of used entries in aCol[] */
        public int nAccumulator;    /* Number of columns that show through to the output. Additional columns are used only as parameters to aggregate functions */
        public Func aFunc;
        public int nFunc;           /* Number of entries in aFunc[] */
    }
}
