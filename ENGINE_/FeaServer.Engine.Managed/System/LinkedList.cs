#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.InteropServices;

namespace System
{
    public interface ILinkedListFirst<T>
        where T : class { T First { get; } }
    public class LinkedList<T> : IEnumerable<T>, IEnumerable, ILinkedListFirst<T>
        where T : LinkedListNode<T, LinkedList<T>>
    {
        internal int count;
        internal T head;
        internal int version;

        public void AddAfter(T node, T newNode)
        {
            ValidateNode(node);
            ValidateNewNode(newNode);
            InternalInsertNodeBefore(node.next, newNode);
            newNode.list = this;
        }

        public void AddBefore(T node, T newNode)
        {
            ValidateNode(node);
            ValidateNewNode(newNode);
            InternalInsertNodeBefore(node, newNode);
            newNode.list = this;
            if (node == head)
                head = newNode;
        }

        public void AddFirst(T node)
        {
            ValidateNewNode(node);
            if (head == null)
                InternalInsertNodeToEmptyList(node);
            else
            {
                InternalInsertNodeBefore(head, node);
                head = node;
            }
            node.list = this;
        }

        public void AddLast(T node)
        {
            ValidateNewNode(node);
            if (head == null)
                InternalInsertNodeToEmptyList(node);
            else
                InternalInsertNodeBefore(head, node);
            node.list = this;
        }

        public void Clear()
        {
            var head = this.head;
            while (head != null)
            {
                var node2 = head;
                head = head.Next;
                node2.Invalidate();
            }
            head = null;
            count = 0;
            version++;
        }

        private void InternalInsertNodeBefore(T node, T newNode)
        {
            newNode.next = node;
            newNode.prev = node.prev;
            node.prev.next = newNode;
            node.prev = newNode;
            version++;
            count++;
        }

        private void InternalInsertNodeToEmptyList(T newNode)
        {
            newNode.next = newNode;
            newNode.prev = newNode;
            head = newNode;
            version++;
            count++;
        }

        internal void InternalRemoveNode(T node)
        {
            if (node.next == node)
                head = null;
            else
            {
                node.next.prev = node.prev;
                node.prev.next = node.next;
                if (head == node)
                    head = node.next;
            }
            node.Invalidate();
            count--;
            version++;
        }

        public void Remove(T node)
        {
            ValidateNode(node);
            InternalRemoveNode(node);
        }

        public void RemoveFirst()
        {
            if (head == null)
                throw new InvalidOperationException("LinkedListEmpty");
            InternalRemoveNode(head);
        }

        public void RemoveLast()
        {
            if (head == null)
                throw new InvalidOperationException("LinkedListEmpty");
            InternalRemoveNode(head.prev);
        }

        internal void ValidateNewNode(T node)
        {
            if (node == null)
                throw new ArgumentNullException("node");
            if (node.list != null)
                throw new InvalidOperationException("LinkedListNodeIsAttached");
        }

        internal void ValidateNode(T node)
        {
            if (node == null)
                throw new ArgumentNullException("node");
            if (node.list != this)
                throw new InvalidOperationException("ExternalLinkedListNode");
        }

        public int Count
        {
            get { return count; }
        }

        public T First
        {
            get { return head; }
        }

        public T Last
        {
            get { return (head != null ? head.prev : null); }
        }

        #region Enumerator

        [StructLayout(LayoutKind.Sequential)]
        public struct Enumerator : IEnumerator<T>, IDisposable, IEnumerator
        {
            private LinkedList<T> list;
            private T node;
            private int version;
            private T current;
            private int index;

            internal Enumerator(LinkedList<T> list)
            {
                this.list = list;
                version = list.version;
                node = list.head;
                current = null;
                index = 0;
            }
            public void Dispose() { }

            public T Current
            {
                get { return current; }
            }
            object IEnumerator.Current
            {
                get
                {
                    if ((index == 0) || (index == (list.Count + 1)))
                        throw new InvalidOperationException("InvalidOperation_EnumOpCantHappen");
                    return current;
                }
            }

            public bool MoveNext()
            {
                if (version != list.version)
                    throw new InvalidOperationException("InvalidOperation_EnumFailedVersion");
                if (node == null)
                {
                    index = list.Count + 1;
                    return false;
                }
                index++;
                current = node;
                node = node.next;
                if (node == list.head)
                    node = null;
                return true;
            }

            void IEnumerator.Reset()
            {
                if (version != list.version)
                    throw new InvalidOperationException("InvalidOperation_EnumFailedVersion");
                current = null;
                node = list.head;
                index = 0;
            }
        }

        IEnumerator<T> IEnumerable<T>.GetEnumerator() { return new Enumerator(this); }
        IEnumerator IEnumerable.GetEnumerator() { return new Enumerator(this); }

        #endregion
    }
}
