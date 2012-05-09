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
using System;
namespace FeaServer.Engine.Core
{
    public class PagerSavepoint
    {
        public long iOffset;            // Starting offset in main journal
        public long iHdrOffset;         // See above
        public Bitvec pInSavepoint;     // Set of pages in this savepoint
        public uint nOrig;              // Original number of pages in file
        public uint iSubRec;            // Index of first record in sub-journal
        //#if !SQLITE_OMIT_WAL
        //public uint aWalData[WAL_SAVEPOINT_NDATA];        // WAL savepoint context
        //#else
        public object aWalData = null;      // Used for C# convenience
        //#endif

        public static implicit operator bool(PagerSavepoint b)
        {
            return (b != null);
        }
    }
}
