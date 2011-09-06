using System;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct Hibernate
    {
        public ElementCollection Elements;

        public Hibernate xtor()
        {
            Console.WriteLine("Hibernate:xtor");
            Elements.xtor();
            return this;
        }
    }
}
