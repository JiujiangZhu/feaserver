namespace FeaServer.Engine.Query
{
    public class SubProgram
    {
        public VdbeOp[] aOp;          /* Array of opcodes for sub-program */
        public int nOp;               /* Elements in aOp[] */
        public int nMem;              /* Number of memory cells required */
        public int nCsr;              /* Number of cursors required */
        public int token;             /* id that may be used to recursive triggers */
        public SubProgram pNext;      /* Next sub-program already visited */
    }
}