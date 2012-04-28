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
using FeaServer.Engine.System;
namespace FeaServer.Engine.Core
{
    public class PGroup
    {
        public sqlite3_mutex mutex;           // MUTEX_STATIC_LRU or NULL
        public int nMaxPage;                  // Sum of nMax for purgeable caches
        public int nMinPage;                  // Sum of nMin for purgeable caches
        public int mxPinned;                  // nMaxpage + 10 - nMinPage
        public int nCurrentPage;              // Number of purgeable pages allocated
        public PgHdr1 pLruHead, pLruTail;     // LRU list of unpinned pages
        
        public PGroup()
        {
            mutex = new sqlite3_mutex();
        }
    }
}
