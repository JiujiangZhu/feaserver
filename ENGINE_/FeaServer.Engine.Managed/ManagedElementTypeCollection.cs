using System;
namespace FeaServer.Engine
{
    public class ManagedElementTypeCollection : ElementTypeCollection
    {
        protected override void InsertItem(int index, IElementType item)
        {
            Console.WriteLine("Managed::InsertItem");
            base.InsertItem(index, item);
        }

        protected override void SetItem(int index, IElementType item)
        {
            throw new NotSupportedException();
        }
    }
}
