using ynVar = System.Int16;
using yDbMask = System.Int32;
using System.IO;
using FeaServer.Engine.Core;
namespace FeaServer.Engine.Query
{
    public class Vdbe
    {
        public enum MAGIC : uint
        {
            INIT = 0x26bceaa5,      // Building a VDBE program
            RUN = 0xbdf20da3,       // VDBE is ready to execute
            HALT = 0x519c2973,      // VDBE has completed execution
            DEAD = 0xb606c3c8,      // The VDBE has been deallocated
        }

        const int COLNAME_NAME = 0;
        const int COLNAME_DECLTYPE = 1;
        const int COLNAME_DATABASE = 2;
        const int COLNAME_TABLE = 3;
        const int COLNAME_COLUMN = 4;
        //const int COLNAME_N = 5;
        //const int COLNAME_N = 1;
        //const int COLNAME_N = 2;

        static void sqlite3VdbeEnter(Vdbe p) { }
        static void sqlite3VdbeLeave(Vdbe p) { }

        public sqlite3 db;              // The database connection that owns this statement 
        public VdbeOp[] aOp;            // Space to hold the virtual machine's program 
        public Mem[] aMem;              // The memory locations 
        public Mem[] apArg;             // Arguments to currently executing user function 
        public Mem[] aColName;          // Column names to return 
        public Mem[] pResultSet;        // Pointer to an array of results 
        public int nMem;                // Number of memory locations currently allocated 
        public int nOp;                 // Number of instructions in the program 
        public int nOpAlloc;            // Number of slots allocated for aOp[] 
        public int nLabel;              // Number of labels used 
        public int nLabelAlloc;         // Number of slots allocated in aLabel[] 
        public int[] aLabel;            // Space to hold the labels 
        public ushort nResColumn;       // Number of columns in one row of the result set 
        public ushort nCursor;          // Number of slots in apCsr[] 
        public MAGIC magic;             // Magic number for sanity checking 
        public string zErrMsg;          // Error message written here 
        public Vdbe pPrev;              // Linked list of VDBEs with the same Vdbe.db 
        public Vdbe pNext;              // Linked list of VDBEs with the same Vdbe.db 
        public VdbeCursor[] apCsr;      // One element of this array for each open cursor 
        public Mem[] aVar;              // Values for the OP_Variable opcode. 
        public string[] azVar;          // Name of variables 
        public ynVar nVar;              // Number of entries in aVar[] 
        public ynVar nzVar;             // Number of entries in azVar[] 
        public uint cacheCtr;           // VdbeCursor row cache generation counter 
        public int pc;                  // The program counter 
        public int rc;                  // Value to return 
        public byte errorAction;        // Recovery action to do in case of an error 
        public int explain;             // True if EXPLAIN present on SQL command 
        public bool changeCntOn;        // True to update the change-counter 
        public bool expired;            // True if the VM needs to be recompiled 
        public byte runOnlyOnce;        // Automatically expire on reset
        public int minWriteFileFormat;  // Minimum file format for writable database files
        public int inVtabMethod;        // See comments above
        public bool usesStmtJournal;    // True if uses a statement journal
        public bool readOnly;           // True for read-only statements
        public int nChange;             // Number of db changes made since last reset
        public bool isPrepareV2;        // True if prepared with prepare_v2()
        public yDbMask btreeMask;       // Bitmask of db.aDb[] entries referenced
        public yDbMask lockMask;        // Subset of btreeMask that requires a lock
        //
        public int iStatement;          // Statement number (or 0 if has not opened stmt)
        public int[] aCounter = new int[3]; // Counters used by sqlite3_stmt_status()
#if !SQLITE_OMIT_TRACE
        public long startTime;          // Time when query started - used for profiling
#endif
        public long nFkConstraint;      // Number of imm. FK constraints this VM
        public long nStmtDefCons;       // Number of def. constraints when stmt started
        public string zSql = "";        // Text of the SQL statement that generated this
        public object pFree;            // Free this when deleting the vdbe
#if DEBUG
        public TextWriter trace;        // Write an execution trace here, if not NULL
#endif
        public VdbeFrame pFrame;        // Parent frame
        public VdbeFrame pDelFrame;     // List of frame objects to free on VM reset
        public int nFrame;              // Number of frames in pFrame list
        public uint expmask;            // Binding to these vars invalidates VM
        public SubProgram pProgram;     // Linked list of all sub-programs used by VM

        public Vdbe Copy()
        {
            var cp = (Vdbe)MemberwiseClone();
            return cp;
        }

        public void CopyTo(Vdbe ct)
        {
            ct.db = db;
            ct.pPrev = pPrev;
            ct.pNext = pNext;
            ct.nOp = nOp;
            ct.nOpAlloc = nOpAlloc;
            ct.aOp = aOp;
            ct.nLabel = nLabel;
            ct.nLabelAlloc = nLabelAlloc;
            ct.aLabel = aLabel;
            ct.apArg = apArg;
            ct.aColName = aColName;
            ct.nCursor = nCursor;
            ct.apCsr = apCsr;
            ct.aVar = aVar;
            ct.azVar = azVar;
            ct.nVar = nVar;
            ct.nzVar = nzVar;
            ct.magic = magic;
            ct.nMem = nMem;
            ct.aMem = aMem;
            ct.cacheCtr = cacheCtr;
            ct.pc = pc;
            ct.rc = rc;
            ct.errorAction = errorAction;
            ct.nResColumn = nResColumn;
            ct.zErrMsg = zErrMsg;
            ct.pResultSet = pResultSet;
            ct.explain = explain;
            ct.changeCntOn = changeCntOn;
            ct.expired = expired;
            ct.minWriteFileFormat = minWriteFileFormat;
            ct.inVtabMethod = inVtabMethod;
            ct.usesStmtJournal = usesStmtJournal;
            ct.readOnly = readOnly;
            ct.nChange = nChange;
            ct.isPrepareV2 = isPrepareV2;
#if !SQLITE_OMIT_TRACE
            ct.startTime = startTime;
#endif
            ct.btreeMask = btreeMask;
            ct.lockMask = lockMask;
            aCounter.CopyTo(ct.aCounter, 0);
            ct.zSql = zSql;
            ct.pFree = pFree;
#if SQLITE_DEBUG
        ct.trace = trace;
#endif
            ct.nFkConstraint = nFkConstraint;
            ct.nStmtDefCons = nStmtDefCons;
            ct.iStatement = iStatement;
            ct.pFrame = pFrame;
            ct.nFrame = nFrame;
            ct.expmask = expmask;
            ct.pProgram = pProgram;
#if SQLITE_SSE
ct.fetchId=fetchId;
ct.lru=lru;
#endif
#if SQLITE_ENABLE_MEMORY_MANAGEMENT
ct.pLruPrev=pLruPrev;
ct.pLruNext=pLruNext;
#endif
        }
    }
}