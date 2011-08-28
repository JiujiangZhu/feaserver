using System.Collections.Generic;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class SliceFractionCollection : SortedDictionary<ulong, SliceNode>
    {
        internal void Schedule(Element element, ulong fraction)
        {
            SliceNode node;
            if (!TryGetValue(fraction, out node))
                Add(fraction, node = new SliceNode(0));
            node.Elements.Add(element, null);
        }
    }
}
