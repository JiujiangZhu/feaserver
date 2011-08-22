using System;
namespace Example
{
    class Program
    {
        static void Main(string[] args)
        {
            using (var c = new TrackerServiceCallback())
            {
                Console.WriteLine("Press <ENTER> to terminate example.");
                Console.ReadKey();                
            }
        }
    }
}
