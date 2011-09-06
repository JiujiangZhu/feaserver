using System;
using System.Collections.Generic;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class SliceFractionCollection : SortedDictionary<ulong, SliceFraction>
    {
        public SliceFractionCollection()
        {
            Console.WriteLine("SliceFractionCollection:ctor");
        }

        internal void Schedule(Element element, ulong fraction)
        {
            Console.WriteLine("SliceFractionCollection:Schedule {0}", TimePrec.DecodeTime(fraction));
            SliceFraction fraction2;
            if (!TryGetValue(fraction, out fraction2))
                Add(fraction, fraction2.xtor());
            fraction2.Elements.Add(element, 0);
        }
    }
}
