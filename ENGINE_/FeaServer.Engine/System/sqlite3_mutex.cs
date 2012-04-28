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
namespace FeaServer.Engine.System
{
    public class sqlite3_mutex
    {
        public object mutex;    /* Mutex controlling the lock */
        public int id;          /* Mutex type */
        public int nRef;        /* Number of enterances */
        public int owner;       /* Thread holding this mutex */
#if DEBUG
        public int trace;       /* True to trace changes */
#endif

        public sqlite3_mutex()
        {
            mutex = new object();
        }

        public sqlite3_mutex(sqlite3_mutex mutex, int id, int nRef, int owner
#if DEBUG
, int trace
#endif
)
        {
            this.mutex = mutex;
            this.id = id;
            this.nRef = nRef;
            this.owner = owner;
#if DEBUG
            this.trace = 0;
#endif
        }
    }
}
