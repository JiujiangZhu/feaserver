using System;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class ElementList : LinkedList<Element>
    {
        public void MergeFirstWins(Element element, byte[] metadata)
        {
            Console.WriteLine("ElementList:MergeFirstWins");
        }

        public void MergeLastWins(Element element, byte[] metadata)
        {
            Console.WriteLine("ElementList:MergeLastWins");
        }
    }
}
