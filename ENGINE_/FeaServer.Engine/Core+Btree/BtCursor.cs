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
    public class BtCursor
    {
        const int BTCURSOR_MAX_DEPTH = 20;

        public enum CURSOR : int
        {
            INVALID = 0,
            VALID = 1,
            REQUIRESEEK = 2,
            FAULT = 3,
        }

        public Btree pBtree;                // The Btree to which this cursor belongs
        public BtShared pBt;                // The BtShared this cursor points to
        public BtCursor pNext;
        public BtCursor pPrev;              // Forms a linked list of all cursors
        public KeyInfo pKeyInfo;            // Argument passed to comparison function
        public uint pgnoRoot;               // The root page of this tree
        public long cachedRowid;            // Next rowid cache.  0 means not valid
        public CellInfo info = new CellInfo();      // A parse of the cell we are pointing at
        public byte[] pKey;                 // Saved key that was cursor's last known position
        public long nKey;                   // Size of pKey, or last integer key
        public int skipNext;                // Prev() is noop if negative. Next() is noop if positive
        public byte wrFlag;                 // True if writable
        public byte atLast;                 // VdbeCursor pointing to the last entry
        public bool validNKey;              // True if info.nKey is valid
        public CURSOR eState;               // One of the CURSOR_XXX constants (see below)
#if !SQLITE_OMIT_INCRBLOB
        public uint[] aOverflow;            // Cache of overflow page locations
        public bool isIncrblobHandle;       // True if this cursor is an incr. io handle
#endif
        public short iPage;                                         // Index of current page in apPage
        public ushort[] aiIdx = new ushort[BTCURSOR_MAX_DEPTH];     // Current index in apPage[i]
        public MemPage[] apPage = new MemPage[BTCURSOR_MAX_DEPTH];  // Pages from root to current page

        public void Clear()
        {
            pNext = null;
            pPrev = null;
            pKeyInfo = null;
            pgnoRoot = 0;
            cachedRowid = 0;
            info = new CellInfo();
            wrFlag = 0;
            atLast = 0;
            validNKey = false;
            eState = 0;
            pKey = null;
            nKey = 0;
            skipNext = 0;
#if !SQLITE_OMIT_INCRBLOB
            isIncrblobHandle = false;
            aOverflow = null;
#endif
            iPage = 0;
        }

        public BtCursor Copy()
        {
            var cp = (BtCursor)MemberwiseClone();
            return cp;
        }
    }
}
