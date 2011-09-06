using System;
using FeaServer.Engine.Time.Scheduler;
namespace FeaServer.Engine
{
    public class Program
    {
        static void Main(string[] args)
        {
            Element e = new Element(null) { ScheduleStyle = ElementScheduleStyle.Multiple };

            var s = new SliceCollection();
            s.Schedule(e, 10);
            s.MoveNextSlice();

            Console.WriteLine("Done.");
        }
    }
}
