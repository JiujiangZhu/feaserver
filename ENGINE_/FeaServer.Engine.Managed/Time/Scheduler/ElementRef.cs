using System;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class ElementRef : LinkedListNode<ElementRef, LinkedList<ElementRef>>
    {
        public Element Element;
        public byte[] Metadata = new byte[Element.MetadataSize]; 
    }
}
