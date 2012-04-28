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
    // Each token coming out of the lexer is an instance of this structure.  Tokens are also used as part of an expression.
    // Note if Token.z==0 then Token.dyn and Token.n are undefined and may contain random values.  Do not make any assumptions about Token.dyn and Token.n when Token.z==0.
    public class Token
    {
        public string z;    // Text of the token.  Not NULL-terminated!
        public int n;       // Number of characters in this token

        public Token()
        {
            z = null;
            n = 0;
        }
        public Token(string z, int n)
        {
            this.z = z;
            this.n = n;
        }

        public Token Copy()
        {
            if (this == null)
                return null;
            else
            {
                var cp = (Token)MemberwiseClone();
                if (z == null || z.Length == 0)
                    cp.n = 0;
                else
                    if (n > z.Length)
                        cp.n = z.Length;
                return cp;
            }
        }
    }
}
