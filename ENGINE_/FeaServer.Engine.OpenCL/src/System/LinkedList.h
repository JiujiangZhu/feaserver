#pragma region License
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
#pragma endregion
#pragma once

namespace System {
	template <typename T>
	class LinkedList;

	template <typename T>
	class LinkedListNode
	{
	public:
        LinkedList<T>* list;
        T* next;
        T* prev;

        __forceinline __device__ void Invalidate()
        {
            list = nullptr;
            next = nullptr;
            prev = nullptr;
        }

        __forceinline __device__ LinkedList<T>* getList()
        {
			trace(LinkedListNode, "getList");
            return list;
        }

        __device__ T* getNext();
        __device__ T* getPrevious();
	};

	template <typename T>
	class LinkedList
	{
	public:
        int count;
        T* head;
        int version;

		__device__ void AddAfter(T* node, T* newNode)
        {
            ValidateNode(node);
            ValidateNewNode(newNode);
            InternalInsertNodeBefore(node->next, newNode);
            newNode->list = this;
        }

        __device__ void AddBefore(T* node, T* newNode)
        {
            ValidateNode(node);
            ValidateNewNode(newNode);
            InternalInsertNodeBefore(node, newNode);
            newNode->list = this;
            if (node == head)
                head = newNode;
        }

        __device__ void AddFirst(T* node)
        {
            ValidateNewNode(node);
            if (head == nullptr)
                InternalInsertNodeToEmptyList(node);
            else
            {
                InternalInsertNodeBefore(head, node);
                head = node;
            }
            node->list = this;
        }

        __device__ void AddLast(T* node)
        {
            ValidateNewNode(node);
            if (head == nullptr)
                InternalInsertNodeToEmptyList(node);
            else
                InternalInsertNodeBefore(head, node);
            node->list = this;
        }

        __device__ void Clear()
        {
            T* head = this->head;
            while (head != nullptr)
            {
                T* node2 = head;
                head = head->getNext();
                node2->Invalidate();
            }
            head = nullptr;
            count = 0;
            version++;
        }

        __device__ void InternalInsertNodeBefore(T* node, T* newNode)
        {
            newNode->next = node;
            newNode->prev = node->prev;
            node->prev->next = newNode;
            node->prev = newNode;
            version++;
            count++;
        }

        __device__ void InternalInsertNodeToEmptyList(T* newNode)
        {
            newNode->next = newNode;
            newNode->prev = newNode;
            head = newNode;
            version++;
            count++;
        }

        __device__ void InternalRemoveNode(T* node)
        {
            if (node->next == node)
                head = nullptr;
            else
            {
                node->next->prev = node->prev;
                node->prev->next = node->next;
                if (head == node)
                    head = node->next;
            }
            node->Invalidate();
            count--;
            version++;
        }

        __device__ void Remove(T* node)
        {
            ValidateNode(node);
            InternalRemoveNode(node);
        }

        __device__ void RemoveFirst()
        {
            if (head == nullptr)
                thrownew(InvalidOperationException, "LinkedListEmpty");
            InternalRemoveNode(head);
        }

        __device__ void RemoveLast()
        {
            if (head == nullptr)
                thrownew(InvalidOperationException, "LinkedListEmpty");
            InternalRemoveNode(head->prev);
        }

        __device__ void ValidateNewNode(T* node)
        {
            if (node == nullptr)
                thrownew(ArgumentNullException, "node");
            if (node->list != nullptr)
                thrownew(InvalidOperationException, "LinkedListNodeIsAttached");
        }

        __device__ void ValidateNode(T* node)
        {
            if (node == nullptr)
                thrownew(ArgumentNullException, "node");
            if (node->list != this)
                thrownew(InvalidOperationException, "ExternalLinkedListNode");
        }

        __device__ int getCount()
        {
            return count;
        }

        __device__ T* getFirst()
        {
            return head;
        }

        __device__ T* getLast()
        {
            return (head != nullptr ? head->prev : nullptr);
        }

		#pragma region Enumerator

        typedef struct Enumerator_t
        {
		public:
            LinkedList<T>* list;
            T* node;
            int version;
            T* Current;
            int index;

            __device__ void xtor(LinkedList<T>* list)
            {
				trace(LinkedList_Enumerator, "xtor");
                this->list = list;
                version = list->version;
                node = list->head;
                Current = nullptr;
                index = 0;
            }

            __forceinline __device__ void Dispose() { }
      
            __device__ bool MoveNext()
            {
                if (version != list->version)
                    thrownew(InvalidOperationException, "InvalidOperation_EnumFailedVersion");
                if (node == nullptr)
                {
                    index = list->getCount() + 1;
                    return false;
                }
                index++;
                Current = node;
                node = node->next;
                if (node == list->head)
                    node = nullptr;
                return true;
            }

            __device__ void Reset()
            {
                if (version != list->version)
                    thrownew(InvalidOperationException, "InvalidOperation_EnumFailedVersion");
                Current = nullptr;
                node = list->head;
                index = 0;
            }
		} Enumerator;

		__device__ bool GetEnumerator(Enumerator& t)
        {
			trace(LinkedList_Enumerator, "GetEnumerator");
			t.xtor(this);
			return t.MoveNext();
        }

        #pragma endregion
	};

	template <typename T>
	__device__ T* LinkedListNode<T>::getNext()
	{
		trace(ElementRef, "getNext");
		return ((next != nullptr) && (next != list->head) ? next : nullptr);
	}

	template <typename T>
	__device__ T* LinkedListNode<T>::getPrevious()
	{
		trace(ElementRef, "getPrevious");
		return ((prev != nullptr) && (this != list->head) ? prev : nullptr);
	}

}