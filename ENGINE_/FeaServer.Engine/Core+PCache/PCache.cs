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
    public class PCache
    {
        public PgHdr pDirty;        // List of dirty pages in LRU order
        public PgHdr pDirtyTail;    
        public PgHdr pSynced;       // Last synced page in dirty page list
        public int nRef;            // Number of referenced pages
        public int nMax;            // Configured cache size
        public int szPage;          // Size of every page in this cache
        public int szExtra;         // Size of extra space for each page
        public bool bPurgeable;     // True if pages are on backing store
        public Func<object, PgHdr, int> xStress;   // Call to try make a page clean
        public object pStress;      // Argument to xStress
        public PCache1 pCache;      // Pluggable cache module
        public PgHdr pPage1;        // Reference to page 1

        public void Clear()
        {
            pDirty = null;
            pDirtyTail = null;
            pSynced = null;
            nRef = 0;
        }
    }
}
