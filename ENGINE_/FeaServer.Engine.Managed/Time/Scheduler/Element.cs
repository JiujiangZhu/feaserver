using System;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class Element
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

        #region LinkedList

        internal ElementList list;
        internal Element next;
        internal Element prev;

        internal void Invalidate()
        {
            list = null;
            next = null;
            prev = null;
        }

        public ElementList List
        {
            get { return list; }
        }

        public Element Next
        {
            get { return ((next != null) && (next != list.head) ? next : null); }
        }

        public Element Previous
        {
            get { return ((prev != null) && (this != list.head) ? prev : null); }
        }

        #endregion
    }
}
