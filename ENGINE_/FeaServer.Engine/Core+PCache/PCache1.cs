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
    public class PCache1
    {
        // Cache configuration parameters. Page size (szPage) and the purgeable flag (bPurgeable) are set when the cache is created. nMax may be 
        // modified at any time by a call to the pcache1CacheSize() method. The PGroup mutex must be held when accessing nMax.
        public PGroup pGroup;           // PGroup this cache belongs to
        public int szPage;              // Size of allocated pages in bytes
        public bool bPurgeable;         // True if cache is purgeable
        public int nMin;                // Minimum number of pages reserved
        public int nMax;                // Configured "cache_size" value
        public int n90pct;              // nMax*9/10
        // Hash table of all pages. The following variables may only be accessed when the accessor is holding the PGroup mutex.
        public int nRecyclable;         // Number of pages in the LRU list
        public int nPage;               // Total number of pages in apHash
        public int nHash;               // Number of slots in apHash[]
        public PgHdr1[] apHash;         // Hash table for fast lookup by key
        //
        public uint iMaxKey;            // Largest key seen since xTruncate()

        public void Clear()
        {
            nRecyclable = 0;
            nPage = 0;
            nHash = 0;
            apHash = null;
            iMaxKey = 0;
        }
    }
}
