using System.Diagnostics;
using System;
using FeaServer.Core;
using System.Runtime.InteropServices;
namespace FeaServer.Query
{
    // A sub-routine used to implement a trigger program.
    public class SubProgram
    {
        VdbeOP[] Ops;                  /* Array of opcodes for sub-program */
        //int NOps;                      /* Elements in aOp[] */
        int Mems;                     /* Number of memory cells required */
        int Csrs;                     /* Number of cursors required */
        int Onces;                    /* Number of OP_Once instructions */
        object token;                  /* id that may be used to recursive triggers */
        SubProgram Next;            /* Next sub-program already visited */
    }
}
