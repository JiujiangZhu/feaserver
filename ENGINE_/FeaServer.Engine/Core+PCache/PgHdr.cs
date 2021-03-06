﻿#region License
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
    public class PgHdr
    {
        public enum PGHDR : int
        {
            DIRTY = 0x002, // Page has changed
            NEED_SYNC = 0x004, // Fsync the rollback journal before writing this page to the database
            NEED_READ = 0x008, // Content is unread
            REUSE_UNLIKELY = 0x010, // A hint that reuse is unlikely
            DONT_WRITE = 0x020, // Do not write content to disk
        }

        public byte[] pData;          /* Content of this page */
        public MemPage pExtra;        /* Extra content */
        public PgHdr pDirty;          /* Transient list of dirty pages */
        public uint pgno;             /* The page number for this page */
        public Pager pPager;          /* The pager to which this page belongs */
#if DEBUG
        public int pageHash;          /* Hash of page content */
#endif
        public PGHDR flags;             /* PGHDR flags defined below */

        // Elements above are public.  All that follows is private to pcache.c and should not be accessed by other modules.

        public int nRef;              /* Number of users of this page */
        public PCache pCache;         /* Cache that owns this page */
        public bool CacheAllocated;   /* True, if allocated from cache */

        public PgHdr pDirtyNext;      /* Next element in list of dirty pages */
        public PgHdr pDirtyPrev;      /* Previous element in list of dirty pages */
        public PgHdr1 pPgHdr1;        /* Cache page header this this page */

        public static implicit operator bool(PgHdr b)
        {
            return (b != null);
        }

        public void Clear()
        {
            sqlite3_free(ref this.pData);
            this.pData = null;
            this.pExtra = null;
            this.pDirty = null;
            this.pgno = 0;
            this.pPager = null;
#if DEBUG
            this.pageHash = 0;
#endif
            this.flags = 0;
            this.nRef = 0;
            this.CacheAllocated = false;
            this.pCache = null;
            this.pDirtyNext = null;
            this.pDirtyPrev = null;
            this.pPgHdr1 = null;
        }
    }
}
