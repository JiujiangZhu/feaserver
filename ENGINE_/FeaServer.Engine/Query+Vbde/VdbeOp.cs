using FeaServer.Engine.Core;
namespace FeaServer.Engine.Query
{
    public class VdbeOp
    {
        public delegate void dxDel(ref string pDelArg);

        public enum P4 : int
        {
            NOTUSED = 0,        // The P4 parameter is not used
            DYNAMIC = -1,       // Pointer to a string obtained from sqliteMalloc=();
            STATIC = -2,        // Pointer to a static string
            COLLSEQ = -4,       // P4 is a pointer to a CollSeq structure
            FUNCDEF = -5,       // P4 is a pointer to a FuncDef structure
            KEYINFO = -6,       // P4 is a pointer to a KeyInfo structure
            VDBEFUNC = -7,      // P4 is a pointer to a VdbeFunc structure
            MEM = -8,           // P4 is a pointer to a Mem*    structure
            TRANSIENT = 0,      // P4 is a pointer to a transient string
            //VTAB = -10,         // P4 is a pointer to an sqlite3_vtab structure
            MPRINTF = -11,      // P4 is a string obtained from sqlite3_mprintf=();
            REAL = -12,         // P4 is a 64-bit floating point value
            INT64 = -13,        // P4 is a 64-bit signed integer
            INT32 = -14,        // P4 is a 32-bit signed integer
            INTARRAY = -15,     // P4 is a vector of 32-bit integers
            SUBPROGRAM = -18,   // P4 is a pointer to a SubProgram structure
            // When adding a P4 argument using P4_KEYINFO, a copy of the KeyInfo structure is made.  That copy is freed when the Vdbe is finalized.  But if the
            // argument is P4_KEYINFO_HANDOFF, the passed in pointer is used.  It still gets freed when the Vdbe is finalized so it still should be obtained
            // from a single sqliteMalloc().  But no copy is made and the calling function should *not* try to free the KeyInfo.
            KEYINFO_HANDOFF = -16,
            KEYINFO_STATIC = -17,
        }

        public byte opcode;             // What operation to perform
        public P4 p4type;               // One of the P4_xxx constants for p4
        public byte opflags;            // Mask of the OPFLG_* flags in opcodes.h
        public byte p5;                 // Fifth parameter is an unsigned character
        public int p1;                  // First operand
        public int p2;                  // Second parameter (often the jump destination)
        public int p3;                  // The third parameter
        public class union_p4
        {   // fourth parameter
            public int i;               // Integer value if p4type==P4_INT32
            public object p;            // Generic pointer
            public string z;            // In C# string is unicode, so use byte[] instead
            public long pI64;            // Used when p4type is P4_INT64
            public double pReal;        // Used when p4type is P4_REAL
            public FuncDef pFunc;       // Used when p4type is P4_FUNCDEF
            public VdbeFunc pVdbeFunc;  // Used when p4type is P4_VDBEFUNC
            public CollSeq pColl;       // Used when p4type is P4_COLLSEQ
            public Mem pMem;            // Used when p4type is P4_MEM
            //public VTable pVtab;        // Used when p4type is P4_VTAB
            public KeyInfo pKeyInfo;    // Used when p4type is P4_KEYINFO
            public int[] ai;            // Used when p4type is P4_INTARRAY
            public SubProgram pProgram; // Used when p4type is P4_SUBPROGRAM
            public dxDel pFuncDel;      // Used when p4type is P4_FUNCDEL
        }
        public union_p4 p4 = new union_p4();
#if DEBUG
        public string zComment;         // Comment to improve readability
#endif
#if VDBE_PROFILE
        public int cnt;             /* Number of times this instruction was executed */
        public u64 cycles;         /* Total time spend executing this instruction */
#endif
    }
}