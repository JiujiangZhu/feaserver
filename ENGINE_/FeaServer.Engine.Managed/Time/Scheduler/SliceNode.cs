using System.Collections.Generic;
namespace FeaServer.Engine.Time.Scheduler
{
    internal struct SliceNode
    {
        public ElementCollection Elements;

        public SliceNode(int none)
        {
            Elements = new ElementCollection(0);
        }
    }
}
