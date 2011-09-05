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

        public ElementTable GetTable(int shard)
        {
            Shard shard2;
            if (!_shards.TryGetValue(shard, out shard2))
                throw new ArgumentNullException("shard");
            return GetTable(shard2);
        }
        private ElementTable GetTable(Shard shard)
        {
            return null;
        }

        public void LoadTable(ElementTable table, int shard)
        {
            Shard shard2;
            if (_shards.TryGetValue(shard, out shard2))
                throw new ArgumentNullException("shard");
            _shards.Add(shard, shard2 = new Shard("Shard: " + shard.ToString()));
            LoadTable(table, shard2);
        }
        private void LoadTable(ElementTable table, Shard shard)
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
