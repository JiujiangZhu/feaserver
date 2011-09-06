using System;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct Slice
    {
        public SliceFractionCollection Fractions;

        public Slice xtor()
        {
            Console.WriteLine("Slice:xtor");
            Fractions = new SliceFractionCollection();
            return this;
        }
    }
}
