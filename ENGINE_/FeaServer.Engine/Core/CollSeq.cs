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
    public class CollSeq
    {
        public delegate void dxDelCollSeq(ref object pDelArg);
        public delegate int dxCompare(object pCompareArg, int size1, string Key1, int size2, string Key2);

        public enum COLL
        {
            BINARY = 1,     // The default memcmp() collating sequence
            NOCASE = 2,     // The built-in NOCASE collating sequence
            REVERSE = 3,    // The built-in REVERSE collating sequence
            USER = 0,       // Any other user-defined collating sequence
        }

        public string zName;        // Name of the collating sequence, UTF-8 encoded
        public byte enc;            // Text encoding handled by xCmp()
        public COLL type;
        public object pUser;        // First argument to xCmp()
        public dxCompare xCmp;
        public dxDelCollSeq xDel;

        public CollSeq Copy()
        {
            if (this == null)
                return null;
            else
            {
                var cp = (CollSeq)MemberwiseClone();
                return cp;
            }
        }
    }
}
