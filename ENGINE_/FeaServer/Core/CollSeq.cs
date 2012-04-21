using System;
namespace FeaServer.Core
{
    /*
    ** A "Collating Sequence" is defined by an instance of the following structure. Conceptually, a collating sequence consists of a name and
    ** a comparison routine that defines the order of that sequence.
    **
    ** There may two separate implementations of the collation function, one that processes text in UTF-8 encoding (CollSeq.xCmp) and another that
    ** processes text encoded in UTF-16 (CollSeq.xCmp16), using the machine native byte order. When a collation sequence is invoked, SQLite selects
    ** the version that will require the least expensive encoding translations, if any.
    **
    ** The CollSeq.pUser member variable is an extra parameter that passed in as the first argument to the UTF-8 comparison function, xCmp.
    ** CollSeq.pUser16 is the equivalent for the UTF-16 comparison function, xCmp16.
    **
    ** If both CollSeq.xCmp and CollSeq.xCmp16 are NULL, it means that the collating sequence is undefined.  Indices built on an undefined
    ** collating sequence may not be read or written.
    */
    public struct CollSeq
    {
        public string Name;         /* Name of the collating sequence, UTF-8 encoded */
        public byte enc;            /* Text encoding handled by xCmp() */
        public object pUser;        /* First argument to xCmp() */
        public Func<object, int, object, int, object, int> xCmp;
        public Action<object> xDel; /* Destructor for pUser */
    }
}
