using System;
using System.Collections.Generic;
namespace FeaServer.Engine
{
	public interface IEngine : IDisposable
	{
        ElementTable GetTable(int shard);
        void LoadTable(ElementTable table, int shard);
        void UnloadTable(int shard);
        ElementTypeCollection Types { get; }
        void EvaluateFrame(ulong time);
	}
}
