using System;
using System.Collections.Generic;
namespace FeaServer.Engine
{
	public interface IEngine : IDisposable
	{
        IEnumerable<IElement> GetElements(int shard);
        void LoadElements(IEnumerable<IElement> elements, int shard);
        ElementTypeCollection ElementTypes { get; }
        void EvaluateFrame(ulong time);
	}
}
