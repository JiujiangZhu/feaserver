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
namespace FeaServer.Engine.Query
{
    public class SrcList
    {
        public enum JT : byte
        {
            INNER = 0x0001,     // Any kind of inner or cross join
            CROSS = 0x0002,     // Explicit use of the CROSS keyword
            NATURAL = 0x0004,   // True for a "natural" join
            LEFT = 0x0008,      // Left outer join
            RIGHT = 0x0010,     // Right outer join
            OUTER = 0x0020,     // The "OUTER" keyword is present
            ERROR = 0x0040,     // unknown or unsupported join type
        }

        public class Item
        {
            public string zDatabase;    // Name of database holding this table
            public string zName;        // Name of the table
            public string zAlias;       // The "B" part of a "A AS B" phrase.  zName is the "A"
            public ITable pTab;          // An SQL table corresponding to zName
            public Select pSelect;      // A SELECT statement used in place of a table name
            public byte isPopulated;    // Temporary table associated with SELECT is populated
            public JT jointype;         // Type of join between this able and the previous
            public byte notIndexed;     // True if there is a NOT INDEXED clause
            public byte iSelectId;      // If pSelect!=0, the id of the sub-select in EQP
            public int iCursor;         // The VDBE cursor number used to access this table
            public Expr pOn;            // The ON clause of a join
            public IdList pUsing;       // The USING clause of a join
            public ulong colUsed;       // Bit N (1<<N) set if column N of pTab is used
            public string zIndex;       // Identifier from "INDEXED BY <zIndex>" clause
            public IIndex pIndex;        // Index structure corresponding to zIndex, if any
        }

        public short nSrc;      // Number of tables or subqueries in the FROM clause
        public short nAlloc;    // Number of entries allocated in a[] below
        public Item[] a;        // One entry for each identifier on the list

        public SrcList Copy()
        {
            if (this == null)
                return null;
            else
            {
                var cp = (SrcList)MemberwiseClone();
                if (a != null) a.CopyTo(cp.a, 0);
                return cp;
            }
        }
    }
}
