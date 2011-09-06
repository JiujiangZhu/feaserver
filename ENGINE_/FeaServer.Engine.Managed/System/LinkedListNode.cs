namespace System
{
    public class LinkedListNode<T, TList>
        where T : class
        where TList : class, ILinkedListFirst<T>
    {
        internal TList list;
        internal T next;
        internal T prev;

        internal void Invalidate()
        {
            list = null;
            next = null;
            prev = null;
        }

        public TList List
        {
            get { return list; }
        }

        public T Next
        {
            get { return ((next != null) && (next != ((ILinkedListFirst<T>)list).First) ? next : null); }
        }

        public T Previous
        {
            get { return ((prev != null) && (this != ((ILinkedListFirst<T>)list).First) ? prev : null); }
        }
    }
}
