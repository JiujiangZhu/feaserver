using System;
using FeaServer.Engine.Core;
namespace FeaServer.Engine.Query
{
    public class Mem
    {
        public delegate void dxDel(ref string pDelArg);

        [Flags]
        public enum MEM : ushort
        {
            Null = 0x0001, // Value is NULL
            Str = 0x0002, // Value is a string
            Int = 0x0004, // Value is an integer
            Real = 0x0008, // Value is a real number
            Blob = 0x0010, // Value is a BLOB
            RowSet = 0x0020, // Value is a RowSet object
            Frame = 0x0040, // Value is a VdbeFrame object
            Invalid = 0x0080, // Value is undefined
            TypeMask = 0x00ff, // Mask of type bits
            //
            Term = 0x0200, // String rep is nul terminated
            Dyn = 0x0400, // Need to call sqliteFree() on Mem.z
            Static = 0x0800, // Mem.z points to a static string
            Ephem = 0x1000, // Mem.z points to an ephemeral string
            Agg = 0x2000, // Mem.z points to an agg function context
            Zero = 0x4000, // Mem.i contains count of 0s appended to blob
        }

        static void MemSetTypeFlag(Mem p, MEM f) { p.flags = p.flags & ~(MEM.TypeMask | MEM.Zero) | f; }
#if DEBUG
        static bool memIsValid(Mem M) { return (M.flags & MEM.Invalid) == 0; }
#else
        static bool memIsValid(Mem M) { return true; }
#endif

        public sqlite3 db;              // The associated database connection
        public string z;                // String value
        public double r;                // Real value
        public struct union_ip
        {
            public long i;              // Integer value used when MEM_Int is set in flags
            public int nZero;           // Used when bit MEM_Zero is set in flags
            public FuncDef pDef;        // Used only when flags==MEM_Agg
            public IRowSet pRowSet;      // Used only when flags==MEM_RowSet
            public VdbeFrame pFrame;    // Used when flags==MEM_Frame
        }
        public union_ip u;
        public byte[] zBLOB;            // BLOB value */
        public int n;                   // Number of characters in string value, excluding '\0'
        public MEM flags;               // Some combination of MEM_Null, MEM_Str, MEM_Dyn, etc.
        public byte type;               // One of SQLITE_NULL, SQLITE_TEXT, SQLITE_INTEGER, etc
        public byte enc;                // SQLITE_UTF8, SQLITE_UTF16BE, SQLITE_UTF16LE
#if DEBUG
        public Mem pScopyFrom;          // This Mem is a shallow copy of pScopyFrom
        public object pFiller;          // So that sizeof(Mem) is a multiple of 8
#endif
        public dxDel xDel;              // If not null, call this function to delete Mem.z
        public Mem _Mem;                // Used when C# overload Z as MEM space
        public SumCtx _SumCtx;          // Used when C# overload Z as Sum context
        public SubProgram[] _SubProgram;// Used when C# overload Z as SubProgram
        public StrAccum _StrAccum;      // Used when C# overload Z as STR context
        public object _MD5Context;      // Used when C# overload Z as MD5 context

        public Mem() { }
        public Mem(sqlite3 db, string z, double r, int i, int n, MEM flags, byte type, byte enc
#if DEBUG
, Mem pScopyFrom, object pFiller
#endif
)
        {
            this.db = db;
            this.z = z;
            this.r = r;
            this.u.i = i;
            this.n = n;
            this.flags = flags;
#if DEBUG
            this.pScopyFrom = pScopyFrom;
            this.pFiller = pFiller;
#endif
            this.type = type;
            this.enc = enc;
        }

        public void CopyTo(ref Mem ct)
        {
            if (ct == null)
                ct = new Mem();
            ct.u = u;
            ct.r = r;
            ct.db = db;
            ct.z = z;
            if (zBLOB == null)
                ct.zBLOB = null;
            else
            {
                ct.zBLOB = new byte[zBLOB.Length];
                Buffer.BlockCopy(zBLOB, 0, ct.zBLOB, 0, zBLOB.Length);
            }
            ct.n = n;
            ct.flags = flags;
            ct.type = type;
            ct.enc = enc;
            ct.xDel = xDel;
        }
    }
}