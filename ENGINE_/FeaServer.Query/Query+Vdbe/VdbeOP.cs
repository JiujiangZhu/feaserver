using System.Runtime.InteropServices;
using FeaServer.Core;
namespace FeaServer.Query
{
    // A single instruction of the virtual machine has an opcode and as many as three operands.  The instruction is recorded as an instance of the following structure:
    public class VdbeOP
    {
        [StructLayout(LayoutKind.Explicit)]
        public struct U
        {
            [FieldOffset(0)]
            public int i;               /* Integer value if p4type==P4_INT32 */
            [FieldOffset(0)]
            public object p;            /* Generic pointer */
            [FieldOffset(0)]
            public string z;            /* Pointer to data for string (char array) types */
            [FieldOffset(0)]
            public long pI64;           /* Used when p4type is P4_INT64 */
            [FieldOffset(0)]
            public double pReal;        /* Used when p4type is P4_REAL */
            [FieldOffset(0)]
            public FuncDef pFunc;       /* Used when p4type is P4_FUNCDEF */
            //[FieldOffset(0)]
            //public VdbeFunc pVdbeFunc;  /* Used when p4type is P4_VDBEFUNC */
            [FieldOffset(0)]
            public CollSeq pColl;       /* Used when p4type is P4_COLLSEQ */
            //[FieldOffset(0)]
            //public Mem pMem;            /* Used when p4type is P4_MEM */
            //[FieldOffset(0)]
            //public VTable pVtab;        /* Used when p4type is P4_VTAB */
            [FieldOffset(0)]
            public KeyInfo pKeyInfo;    /* Used when p4type is P4_KEYINFO */
            [FieldOffset(0)]
            public int ai;              /* Used when p4type is P4_INTARRAY */
            [FieldOffset(0)]
            public SubProgram pProgram; /* Used when p4type is P4_SUBPROGRAM */
            //[FieldOffset(0)]
            //public Func<BtCursor , int, int> xAdvance;
        }
        public byte opcode;     /* What operation to perform */
        public VdbeP4Type p4type;   /* One of the P4_xxx constants for p4 */
        public byte opflags;    /* Mask of the OPFLG_* flags in opcodes.h */
        public byte p5;         /* Fifth parameter is an unsigned character */
        public int p1;          /* First operand */
        public int p2;          /* Second parameter (often the jump destination) */
        public int p3;          /* The third parameter */
        public U p4;            /* fourth parameter */
        public string Comment;  /* Comment to improve readability */
        public int cnt;         /* Number of times this instruction was executed */
        public ulong cycles;    /* Total time spent executing this instruction */
    }
}
