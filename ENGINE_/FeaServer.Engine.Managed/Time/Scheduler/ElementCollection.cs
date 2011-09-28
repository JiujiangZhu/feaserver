#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
using System;
using System.Collections.Generic;
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
