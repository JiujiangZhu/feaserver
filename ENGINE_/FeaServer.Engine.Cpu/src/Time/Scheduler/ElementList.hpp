#pragma once
#include "Element.hpp"
namespace Time { namespace Scheduler {
#ifdef _ELEMENT

#else
	#define ELEMENTLIST
	class ElementList
	{
	public:
		__device__ void MergeFirstWins(Element* element, byte* metadata)
        {
			trace(ElementList, "MergeFirstWins");
        }

        __device__ void MergeLastWins(Element* element, byte* metadata)
        {
			trace(ElementList, "MergeLastWins");
        }

		#pragma region LinkedList

	public:
        int count;
        Element* head;
        int version;

		__device__ void AddAfter(Element* node, Element* newNode)
        {
            ValidateNode(node);
            ValidateNewNode(newNode);
            InternalInsertNodeBefore(node->next, newNode);
            newNode->list = this;
        }

        __device__ void AddBefore(Element* node, Element* newNode)
        {
            ValidateNode(node);
            ValidateNewNode(newNode);
            InternalInsertNodeBefore(node, newNode);
            newNode->list = this;
            if (node == head)
                head = newNode;
        }

        __device__ void AddFirst(Element* node)
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

        __device__ void AddLast(Element* node)
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
            Element* head = this->head;
            while (head != nullptr)
            {
                Element* node2 = head;
                head = head->getNext();
                node2->Invalidate();
            }
            head = nullptr;
            count = 0;
            version++;
        }

        __device__ void InternalInsertNodeBefore(Element* node, Element* newNode)
        {
            newNode->next = node;
            newNode->prev = node->prev;
            node->prev->next = newNode;
            node->prev = newNode;
            version++;
            count++;
        }

        __device__ void InternalInsertNodeToEmptyList(Element* newNode)
        {
            newNode->next = newNode;
            newNode->prev = newNode;
            head = newNode;
            version++;
            count++;
        }

        __device__ void InternalRemoveNode(Element* node)
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

        __device__ void Remove(Element* node)
        {
            ValidateNode(node);
            InternalRemoveNode(node);
        }

        __device__ void RemoveFirst()
        {
            if (head == nullptr)
                throw(InvalidOperationException, "LinkedListEmpty");
            InternalRemoveNode(head);
        }

        __device__ void RemoveLast()
        {
            if (head == nullptr)
                throw(InvalidOperationException, "LinkedListEmpty");
            InternalRemoveNode(head->prev);
        }

        __device__ void ValidateNewNode(Element* node)
        {
            if (node == nullptr)
                throw(ArgumentNullException, "node");
            if (node->list != nullptr)
                throw(InvalidOperationException, "LinkedListNodeIsAttached");
        }

        __device__ void ValidateNode(Element* node)
        {
            if (node == nullptr)
                throw(ArgumentNullException, "node");
            if (node->list != this)
                throw(InvalidOperationException, "ExternalLinkedListNode");
        }

        __device__ int getCount()
        {
            return count;
        }

        __device__ Element* getFirst()
        {
            return head;
        }

        __device__ Element* getLast()
        {
            return (head != nullptr ? head->prev : nullptr);
        }

		#pragma region Enumerator

        typedef struct Enumerator_t
        {
		public:
            ElementList* list;
            Element* node;
            int version;
            Element* Current;
            int index;

            __device__ void xtor(ElementList* list)
            {
				trace(ElementList_Enumerator, "xtor");
                this->list = list;
                version = list->version;
                node = list->head;
                Current = nullptr;
                index = 0;
            }

            __forceinline __device__ void Dispose() { }
      
            /*
			__forceinline __device__ Element* getCurrent()
            {
                return current;
            }
			*/

            __device__ bool MoveNext()
            {
                if (version != list->version)
                    throw(InvalidOperationException, "InvalidOperation_EnumFailedVersion");
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
                    throw(InvalidOperationException, "InvalidOperation_EnumFailedVersion");
                Current = nullptr;
                node = list->head;
                index = 0;
            }
		} Enumerator;

		__device__ bool GetEnumerator(Enumerator& t)
        {
			trace(ElementList_Enumerator, "GetEnumerator");
			t.xtor(this);
			return t.MoveNext();
        }

        #pragma endregion

		#pragma endregion
	};
#endif
}}
#include "Element.hpp"