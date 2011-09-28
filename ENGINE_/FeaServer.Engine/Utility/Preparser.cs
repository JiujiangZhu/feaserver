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
using System.IO;
using System.Text;
namespace FeaServer.Engine.Utility
{
    public static class Preparser
    {
        public static string ReadAllText(string path)
        {
            var b = new StringBuilder();
            Recurse(b, path);
            return b.ToString();
        }

        private static void Recurse(StringBuilder b, string path)
        {
            if (!File.Exists(path))
                throw new Exception("File Not Found: " + path);
            string body = File.ReadAllText(path);
            if (string.IsNullOrEmpty(body))
                return;
            int startIndex = 0;
            do
            {
                var beginOffset = body.IndexOf("#include \"", startIndex);
                if (beginOffset == -1)
                {
                    b.Append(body.Substring(startIndex));
                    break;
                }
                b.Append(body.Substring(startIndex, beginOffset - startIndex));
                //
                beginOffset += 10;
                var endOffset = body.IndexOf("\"", beginOffset);
                //
                var parentPath = Path.GetDirectoryName(path);
                var newPath = ParseInclude(body, beginOffset, endOffset);
                newPath = Path.Combine(parentPath, newPath);
                Recurse(b, newPath);
                startIndex = endOffset + 1;
            } while (startIndex > 0);
        }

        private static string ParseInclude(string body, int beginOffset, int endOffset)
        {
            return body.Substring(beginOffset, endOffset - beginOffset);
        }
    }
}
