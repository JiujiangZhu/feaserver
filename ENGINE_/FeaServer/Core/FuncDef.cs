using System;
namespace FeaServer.Core
{
    /// <summary>
    /// Each SQL function is defined by an instance of the following structure.  A pointer to this structure is stored in the sqlite.aFunc
    /// hash table.  When multiple functions have the same name, the hash table points to a linked list of these structures.
    /// </summary>
    public class FuncDef
    {
        public short Args;                  // Number of arguments.  -1 means unlimited
        public byte PrefEnc;               // Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
        public FuncDefFlags Flags;          // Some combination of FuncDefFlags
        public object UserData;             // User data parameter
        public FuncDef Next;                // Next function with same name
        //public Action<sqlite3_context, int, sqlite3_value[]> xFunc; // Regular function
        //public Action<sqlite3_context, int, sqlite3_value[]> xStep; // Aggregate step
        //public Action<sqlite3_context> xFinalize;                   // Aggregate finalizer
        public string Name;                 // SQL name of the function.
        public FuncDef Hash;               // Next with a different name but the same hash
        public FuncDestructor Destructor;   // Reference counted destructor function
    }
}
