using System.Collections.Generic;
using System;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct ElementCollection
    {
        internal class A { public Element E; public byte[] M = new byte[Element.MetadataSize]; }
        private ElementList _singles;
        private System.LinkedList<ElementRef> _multiples;

        public ElementCollection xtor()
        {
            Console.WriteLine("ElementCollection:xtor");
            _singles = new ElementList();
            _multiples = new System.LinkedList<ElementRef>();
            return this;
        }

        public void Add(Element element, ulong time)
        {
            Console.WriteLine("ElementCollection:Add {0}", TimePrec.DecodeTime(time));
            var metadata = BitConverter.GetBytes(time);
            switch (element.ScheduleStyle)
            {
                case ElementScheduleStyle.FirstWins:
                    _singles.MergeFirstWins(element, metadata);
                    break;
                case ElementScheduleStyle.LastWins:
                    _singles.MergeLastWins(element, metadata);
                    break;
                case ElementScheduleStyle.Multiple:
                    var elementRef = new ElementRef { Element = element, Metadata = metadata };
                    _multiples.AddFirst(elementRef);
                    break;
                default:
                    Console.WriteLine("Warn:UNDEFINED");
                    throw new NotImplementedException();
            }
        }

        public void Clear()
        {
            Console.WriteLine("ElementCollection:Clear");
            _singles.Clear();
            _multiples.Clear();
        }

        public int Count
        {
            get { return _singles.Count + _multiples.Count; }
        }

        public IList<Element> ToList()
        {
            Console.WriteLine("ElementCollection:ToList");
            var list = new List<Element>();
            foreach (var singles in _singles)
                list.Add(singles);
            foreach (var multiple in _multiples)
                list.Add(multiple.Element);
            return list;
        }

        public void DeHibernate(SliceCollection slices)
        {
            Console.WriteLine("ElementCollection:DeHibernate");
            if (_singles.Count > 0)
                foreach (var single in _singles)
                {
                    var time = BitConverter.ToUInt64(single.Metadata, 0);
                    if (time < EngineSettings.MaxTimeslicesTime)
                        throw new Exception("paranoia");
                    var newTime = (ulong)(time -= EngineSettings.MaxTimeslicesTime);
                    if (newTime < EngineSettings.MaxTimeslicesTime)
                    {
                        _singles.Remove(single);
                        slices.Schedule(single, newTime);
                    }
                }
            if (_multiples.Count > 0)
                foreach (var multiple in _multiples)
                {
                    var time = BitConverter.ToUInt64(multiple.Metadata, 0);
                    if (time < EngineSettings.MaxTimeslicesTime)
                        throw new Exception("paranoia");
                    var newTime = (ulong)(time -= EngineSettings.MaxTimeslicesTime);
                    if (newTime < EngineSettings.MaxTimeslicesTime)
                    {
                        _multiples.Remove(multiple);
                        slices.Schedule(multiple.Element, newTime);
                    }
                }
        }
    }
}
