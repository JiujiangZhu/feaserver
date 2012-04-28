using FeaServer.Engine.Core;
namespace FeaServer.Engine.Query
{
    public class sqlite3_context
    {
        public FuncDef pFunc;           // Pointer to function information.  MUST BE FIRST
        public VdbeFunc pVdbeFunc;      // Auxilary data, if created.
        public Mem s = new Mem();       // The return value is stored here
        public Mem pMem;                // Memory cell used to store aggregate context
        public int isError;             // Error code returned by the function.
        public CollSeq pColl;           // Collating sequence
    }
}