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
    public class PgHdr1
    {
        public uint iKey;                   // Key value (page number)
        public PgHdr1 pNext;                // Next in hash table chain
        public PCache1 pCache;              // Cache that currently owns this page
        public PgHdr1 pLruNext;             // Next in LRU list of unpinned pages
        public PgHdr1 pLruPrev;             // Previous in LRU list of unpinned pages

        public PgHdr pPgHdr = new PgHdr();  // Pointer to Actual Page Header

        public void Clear()
        {
            this.iKey = 0;
            this.pNext = null;
            this.pCache = null;
            this.pPgHdr.Clear();
        }
    }
}
