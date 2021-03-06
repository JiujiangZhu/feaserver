using System;
namespace FeaServer.Core
{
    /// <summary>
    /// This structure encapsulates a user-function destructor callback (as configured using create_function_v2()) and a reference counter. When
    /// create_function_v2() is called to create a function with a destructor, a single object of this type is allocated. FuncDestructor.nRef is set to 
    /// the number of FuncDef objects created (either 1 or 3, depending on whether or not the specified encoding is SQLITE_ANY). The FuncDef.pDestructor
    /// member of each of the new FuncDef objects is set to point to the allocated FuncDestructor.
    /// 
    /// Thereafter, when one of the FuncDef objects is deleted, the reference count on this object is decremented. When it reaches 0, the destructor is invoked and the FuncDestructor structure freed.
    /// </summary>
    public class FuncDestructor
    {
        public int Refs;
        public Action<object> Destroy;
        public object UserData;
    }
}
