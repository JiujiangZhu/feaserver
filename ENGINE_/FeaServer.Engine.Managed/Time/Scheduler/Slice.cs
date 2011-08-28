using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace FeaServer.Engine.Time.Scheduler
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct Slice
    {
        public SliceFractionCollection Fractions;

        public Slice(int none)
        {
            Fractions = new SliceFractionCollection();
        }
    }
}
