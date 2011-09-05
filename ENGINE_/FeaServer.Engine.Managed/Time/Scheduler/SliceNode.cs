using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct SliceNode
    {
        public ElementCollection Elements;

        public SliceNode xtor()
        {
            Elements.xtor();
            return this;
        }
    }
}
