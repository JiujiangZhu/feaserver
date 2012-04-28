namespace FeaServer.Engine.Query
{
    public class VdbeFrame
    {
        public Vdbe v;                  // VM this frame belongs to
        public int pc;                  // Program Counter in parent (calling) frame
        public VdbeOp[] aOp;            // Program instructions for parent frame
        public int nOp;                 // Size of aOp array
        public Mem[] aMem;              // Array of memory cells for parent frame
        public int nMem;                // Number of entries in aMem
        public VdbeCursor[] apCsr;      // Array of Vdbe cursors for parent frame
        public ushort nCursor;          // Number of entries in apCsr
        public int token;               // Copy of SubProgram.token
        public int nChildMem;           // Number of memory cells for child frame
        public int nChildCsr;           // Number of cursors for child frame
        public long lastRowid;          // Last insert rowid (sqlite3.lastRowid)
        public int nChange;             // Statement changes (Vdbe.nChanges)
        public VdbeFrame pParent;       // Parent of this frame, or NULL if parent is main
        // Needed for C# Implementation
        public Mem[] aChildMem;         // Array of memory cells for child frame
        public VdbeCursor[] aChildCsr;  // Array of cursors for child frame
    }
}