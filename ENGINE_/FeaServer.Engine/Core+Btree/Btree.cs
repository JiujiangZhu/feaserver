#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
namespace FeaServer.Engine.Core
{
    public class Btree
    {
        public enum TRANS : byte
        {
            NONE = 0,
            READ = 1,
            WRITE = 2,
        }

        public sqlite3 db;        /* The database connection holding this Btree */
        public BtShared pBt;      /* Sharable content of this Btree */
        public TRANS inTrans;        /* TRANS_NONE, TRANS_READ or TRANS_WRITE */
        public bool sharable;     /* True if we can share pBt with another db */
        public bool locked;       /* True if db currently has pBt locked */
        public int wantToLock;    /* Number of nested calls to sqlite3BtreeEnter() */
        public int nBackup;       /* Number of backup operations reading this btree */
        public Btree pNext;       /* List of other sharable Btrees from the same db */
        public Btree pPrev;       /* Back pointer of the same list */
        //#if !SQLITE_OMIT_SHARED_CACHE
        //BtLock lock;              /* Object used to lock page 1 */
        //#endif
    }
}
