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
namespace FeaServer.Engine.Query
{
    public class FuncDef
    {
        public delegate void dxFunc(sqlite3_context ctx, int intValue, Mem[] value);
        public delegate void dxStep(sqlite3_context ctx, int intValue, Mem[] value);
        public delegate void dxFinal(sqlite3_context ctx);
        public delegate void dxFDestroy(object pArg);

        public class FuncDestructor
        {
            public int nRef;
            public dxFDestroy xDestroy;
            public object pUserData;
        }

        [Flags]
        public enum FUNC : byte
        {
            LIKE = 0x01,      // Candidate for the LIKE optimization
            CASE = 0x02,      // Case-sensitive LIKE-type function
            EPHEM = 0x04,     // Ephermeral.  Delete with VDBE
            NEEDCOLL = 0x08,  // sqlite3GetFuncCollSeq() might be called
            PRIVATE = 0x10,   // Allowed for internal use only
            COUNT = 0x20,     // Built-in count() aggregate
            COALESCE = 0x40,  // Built-in coalesce() or ifnull() function
        }

        public short nArg;          // Number of arguments.  -1 means unlimited
        public byte iPrefEnc;       // Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
        public FUNC flags;          // Some combination of SQLITE_FUNC_*
        public object pUserData;    // User data parameter
        public FuncDef pNext;       // Next function with same name
        public dxFunc xFunc;        // Regular function
        public dxStep xStep;        // Aggregate step
        public dxFinal xFinalize;   // Aggregate finalizer
        public string zName;        // SQL name of the function.
        public FuncDef pHash;       // Next with a different name but the same hash
        public FuncDestructor pDestructor;   // Reference counted destructor function

        public FuncDef() { }
        public FuncDef(short nArg, byte iPrefEnc, FUNC iflags, object pUserData, FuncDef pNext, dxFunc xFunc, dxStep xStep, dxFinal xFinalize, string zName, FuncDef pHash, FuncDestructor pDestructor)
        {
            this.nArg = nArg;
            this.iPrefEnc = iPrefEnc;
            this.flags = iflags;
            this.pUserData = pUserData;
            this.pNext = pNext;
            this.xFunc = xFunc;
            this.xStep = xStep;
            this.xFinalize = xFinalize;
            this.zName = zName;
            this.pHash = pHash;
            this.pDestructor = pDestructor;
        }
        public FuncDef(string zName, byte iPrefEnc, short nArg, int iArg, FUNC iflags, dxFunc xFunc)
        {
            this.nArg = nArg;
            this.iPrefEnc = iPrefEnc;
            this.flags = iflags;
            this.pUserData = iArg;
            this.pNext = null;
            this.xFunc = xFunc;
            this.xStep = null;
            this.xFinalize = null;
            this.zName = zName;
        }
        public FuncDef(string zName, byte iPrefEnc, short nArg, int iArg, FUNC iflags, dxStep xStep, dxFinal xFinal)
        {
            this.nArg = nArg;
            this.iPrefEnc = iPrefEnc;
            this.flags = iflags;
            this.pUserData = iArg;
            this.pNext = null;
            this.xFunc = null;
            this.xStep = xStep;
            this.xFinalize = xFinal;
            this.zName = zName;
        }
        public FuncDef(string zName, byte iPrefEnc, short nArg, object arg, dxFunc xFunc, FUNC flags)
        {
            this.nArg = nArg;
            this.iPrefEnc = iPrefEnc;
            this.flags = flags;
            this.pUserData = arg;
            this.pNext = null;
            this.xFunc = xFunc;
            this.xStep = null;
            this.xFinalize = null;
            this.zName = zName;
        }

        public FuncDef Copy()
        {
            var c = new FuncDef();
            c.nArg = nArg;
            c.iPrefEnc = iPrefEnc;
            c.flags = flags;
            c.pUserData = pUserData;
            c.pNext = pNext;
            c.xFunc = xFunc;
            c.xStep = xStep;
            c.xFinalize = xFinalize;
            c.zName = zName;
            c.pHash = pHash;
            c.pDestructor = pDestructor;
            return c;
        }
    }
}
