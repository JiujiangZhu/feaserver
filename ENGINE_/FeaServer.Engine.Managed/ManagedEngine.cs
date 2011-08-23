using System.Collections.Generic;
namespace FeaServer.Engine
{
    public class ManagedEngine : IEngine
    {
        private ElementTypeCollection _elementTypes = new ManagedElementTypeCollection();

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

        public ElementTypeCollection ElementTypes
        {
            get { return _elementTypes; }
        }

        public void EvaluateFrame(ulong time)
        {
        }
    }
}
