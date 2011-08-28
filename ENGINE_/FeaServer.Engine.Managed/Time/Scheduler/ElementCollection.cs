using System.Collections.Generic;
using System;
namespace FeaServer.Engine.Time.Scheduler
{
    internal struct ElementCollection
    {
        internal class A { public Element E; public byte[] M = new byte[Element.MetadataSize]; }
        private LinkedList<Element> _singles;
        private List<A> _multiples;

        public ElementCollection(int none)
        {
            _singles = new LinkedList<Element>();
            _multiples = new List<A>();
        }

        public void Add(Element element, byte[] metadata)
        {
            switch (element.ScheduleStyle)
            {
                case ElementScheduleStyle.FirstWins:
                    break;
                case ElementScheduleStyle.LastWins:
                    break;
                case ElementScheduleStyle.Multiple:
                    _multiples.Add(new A { E = element, M = metadata });
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        //public void Walk(Action<Element, byte[]> action)
        //{
        //    var singles = Singles;
        //    if (singles.Count > 0)
        //        foreach (var single in singles)
        //            action(single, single.Metadata);
        //    var multiples = Multiples;
        //    if (multiples.Count > 0)
        //        foreach (var multiple in multiples)
        //            action(multiple.E, multiple.M);
        //}

        public void WalkAndRemove(Func<Element, byte[], object> predicate, Action<Element, byte[], object> action)
        {
            var singles = _singles;
            if (singles.Count > 0)
                foreach (var single in singles)
                {
                    var value = predicate(single, single.Metadata);
                    if (value != null)
                    {
                        singles.Remove(single);
                        action(single, single.Metadata, value);
                    }
                }
            var multiples = _multiples;
            if (multiples.Count > 0)
                foreach (var multiple in multiples)
                {
                    var value = predicate(multiple.E, multiple.M);
                    if (value != null)
                    {
                        multiples.Remove(multiple);
                        action(multiple.E, multiple.M, value);
                    }
                }
        }
    }
}
