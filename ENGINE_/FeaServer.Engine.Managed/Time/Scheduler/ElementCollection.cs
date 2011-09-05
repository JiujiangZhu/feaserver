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
        private List<A> _multiples;

        public ElementCollection xtor()
        {
            _multiples = new List<A>();
            return this;
        }

        public void Add(Element element, ulong time)
        {
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
                    _multiples.Add(new A { E = element, M = metadata });
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        public void Clear()
        {
            _singles.Clear();
            _multiples.Clear();
        }

        public int Count
        {
            get { return _singles.Count + _multiples.Count; }
        }

        public IList<Element> ToList()
        {
            var list = new List<Element>();
            foreach (var singles in _singles)
                list.Add(singles);
            foreach (var multiple in _multiples)
                list.Add(multiple.E);
            return list;
        }

        public void DeHibernate(SliceCollection slices)
        {
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
                    var time = BitConverter.ToUInt64(multiple.M, 0);
                    if (time < EngineSettings.MaxTimeslicesTime)
                        throw new Exception("paranoia");
                    var newTime = (ulong)(time -= EngineSettings.MaxTimeslicesTime);
                    if (newTime < EngineSettings.MaxTimeslicesTime)
                    {
                        _multiples.Remove(multiple);
                        slices.Schedule(multiple.E, newTime);
                    }
                }
        }
    }
}
