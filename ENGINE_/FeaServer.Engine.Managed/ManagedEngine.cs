using System.Collections.Generic;
using FeaServer.Engine.Time;
using FeaServer.Engine.Time.Scheduler;
namespace FeaServer.Engine
{
    public class ManagedEngine : IEngine
    {
        private ElementTypeCollection _elementTypes = new ManagedElementTypeCollection();
        private SliceCollection _slices = new SliceCollection();

        public ManagedEngine()
        {
        }
        public void Dispose()
        {
        }

        public IEnumerable<IElement> GetElements(int shard)
        {
            return null;
        }

        public void LoadElements(IEnumerable<IElement> elements, int shard)
        {
        }

        public ElementTypeCollection ElementTypes
        {
            get { return _elementTypes; }
        }

        public void EvaluateFrame(ulong time)
        {
        }
    }
}
