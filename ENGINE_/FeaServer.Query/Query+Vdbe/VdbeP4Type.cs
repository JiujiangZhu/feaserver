using System.Runtime.InteropServices;
using FeaServer.Core;
namespace FeaServer.Query
{
    /// <summary>
    /// Allowed values of VdbeOp.p4type
    /// </summary>
    public enum VdbeP4Type : sbyte
    {
        P4_NOTUSED = 0,     /* The P4 parameter is not used */
        P4_DYNAMIC = -1,    /* Pointer to a string obtained from sqliteMalloc() */
        P4_STATIC = -2,     /* Pointer to a static string */
        P4_COLLSEQ = -4,    /* P4 is a pointer to a CollSeq structure */
        P4_FUNCDEF = -5,    /* P4 is a pointer to a FuncDef structure */
        P4_KEYINFO = -6,    /* P4 is a pointer to a KeyInfo structure */
        P4_VDBEFUNC = -7,   /* P4 is a pointer to a VdbeFunc structure */
        P4_MEM = -8,        /* P4 is a pointer to a Mem*    structure */
        P4_TRANSIENT = 0,   /* P4 is a pointer to a transient string */
        P4_VTAB = -10,      /* P4 is a pointer to an sqlite3_vtab structure */
        P4_MPRINTF = -11,   /* P4 is a string obtained from sqlite3_mprintf() */
        P4_REAL = -12,      /* P4 is a 64-bit floating point value */
        P4_INT64 = -13,     /* P4 is a 64-bit signed integer */
        P4_INT32 = -14,     /* P4 is a 32-bit signed integer */
        P4_INTARRAY = -15,  /* P4 is a vector of 32-bit integers */
        P4_SUBPROGRAM = -18,/* P4 is a pointer to a SubProgram structure */
        P4_ADVANCE = -19,   /* P4 is a pointer to BtreeNext() or BtreePrev() */
        // When adding a P4 argument using P4_KEYINFO, a copy of the KeyInfo structure is made.  That copy is freed when the Vdbe is finalized.  But if the
        // argument is P4_KEYINFO_HANDOFF, the passed in pointer is used.  It still gets freed when the Vdbe is finalized so it still should be obtained
        // from a single sqliteMalloc().  But no copy is made and the calling function should *not* try to free the KeyInfo.
        P4_KEYINFO_HANDOFF = -16,
        P4_KEYINFO_STATIC = -17,
    }
}
