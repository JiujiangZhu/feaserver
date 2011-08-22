using System;
using System.Collections.Generic;
namespace FeaServer.Engine
{
	public interface IEngine : IDisposable
	{
        IEnumerable<IElement> GetElements(int shard);
        ElementTypeCollection ElementTypes { get; }
        void EvaluateFrame(ulong time);
	}
}
