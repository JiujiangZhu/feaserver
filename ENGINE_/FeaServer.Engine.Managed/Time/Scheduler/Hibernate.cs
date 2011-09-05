using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct Hibernate
    {
        public ElementCollection Elements;

        public Hibernate xtor()
        {
            Elements.xtor();
            return this;
        }
    }
}
