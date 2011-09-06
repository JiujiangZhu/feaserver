using System;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct SliceFraction
    {
        public ElementCollection Elements;

        public SliceFraction xtor()
        {
            Console.WriteLine("SliceFraction:xtor");
            Elements.xtor();
            return this;
        }
    }
}
