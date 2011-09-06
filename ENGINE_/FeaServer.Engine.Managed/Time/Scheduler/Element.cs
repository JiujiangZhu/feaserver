using System;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class Element : LinkedListNode<Element, LinkedList<Element>>
    {
        internal const int MetadataSize = 4;
        public ElementScheduleStyle ScheduleStyle;
        internal readonly byte[] Metadata = new byte[MetadataSize];
        public IElement Item;

        public Element(IElement element)
        {
            ScheduleStyle = ElementScheduleStyle.FirstWins;
            Item = element;
        }
    }
}
