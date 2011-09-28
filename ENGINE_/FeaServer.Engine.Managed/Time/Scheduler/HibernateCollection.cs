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
    internal struct HibernateCollection
    {
        private Hibernate[] _hibernates;

        public HibernateCollection xtor()
        {
            Console.WriteLine("HibernateCollection:xtor");
            _hibernates = new Hibernate[EngineSettings.MaxHibernates];
            for (int hibernateIndex = 0; hibernateIndex < _hibernates.Length; hibernateIndex++)
                _hibernates[hibernateIndex].xtor();
            return this;
        }
        public void Dispose()
        {
            Console.WriteLine("HibernateCollection:Dispose");
        }

        public void Hibernate(Element element, ulong time)
        {
            Console.WriteLine("HibernateCollection:Hibernate {0}", TimePrec.DecodeTime(time));
            var hibernate = _hibernates[0];
            hibernate.Elements.Add(element, time);
        }

        public void DeHibernate(SliceCollection slices)
        {
            Console.WriteLine("HibernateCollection:DeHibernate");
            var hibernate = _hibernates[0];
            hibernate.Elements.DeHibernate(slices);
        }

        public void ReShuffle()
        {
            Console.WriteLine("HibernateCollection:ReShuffle");
        }
    }
}
