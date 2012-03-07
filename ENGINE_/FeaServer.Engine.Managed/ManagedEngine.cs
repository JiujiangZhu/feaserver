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
using System.Collections.Generic;

namespace FeaServer.Engine
{
    public partial class ManagedEngine : IEngine
    {
        private ElementTypeCollection _types = new ManagedElementTypeCollection();
        private Dictionary<int, Shard> _shards = new Dictionary<int, Shard>();

        public ManagedEngine()
        {
        }
        public void Dispose()
        {
        }

        public CompoundTable GetTable(int shard)
        {
            Shard shard2;
            if (!_shards.TryGetValue(shard, out shard2))
                throw new ArgumentNullException("shard");
            return GetTable(shard2);
        }
        private CompoundTable GetTable(Shard shard)
        {
            return null;
        }

        public void LoadTable(CompoundTable table, int shard)
        {
            Shard shard2;
            if (_shards.TryGetValue(shard, out shard2))
                throw new ArgumentNullException("shard");
            _shards.Add(shard, shard2 = new Shard("Shard: " + shard.ToString()));
            LoadTable(table, shard2);
        }
        private void LoadTable(CompoundTable table, Shard shard)
        {
        }

        public void UnloadTable(int shard)
        {
            Shard shard2;
            if (!_shards.TryGetValue(shard, out shard2))
                throw new ArgumentNullException("shard");
            UnloadTable(shard2);
        }
        private void UnloadTable(Shard shard)
        {
        }

        public ElementTypeCollection Types
        {
            get { return _types; }
        }

        public void EvaluateFrame(ulong time)
        {
        }
    }
}
