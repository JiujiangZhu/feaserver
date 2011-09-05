//#pragma once
namespace Time { namespace Scheduler {
#define MetadataSize 4
#ifdef _ELEMENT

#elif defined(ELEMENTLIST)
	__device__ Element* Element::getNext()
	{
		trace(Element, "getNext");
		return ((next != nullptr) && (next != list->head) ? next : nullptr);
	}

	__device__ Element* Element::getPrevious()
	{
		trace(Element, "getPrevious");
		return ((prev != nullptr) && (this != list->head) ? prev : nullptr);
	}

#else
	#define ELEMENT

	class ElementList;
	class Element
	{

	public:
		ElementScheduleStyle ScheduleStyle;
		byte Metadata[MetadataSize];

	#pragma region LinkedList

	public:
        ElementList* list;
        Element* next;
        Element* prev;

        __forceinline __device__ void Invalidate()
        {
            list = nullptr;
            next = nullptr;
            prev = nullptr;
        }

        __forceinline __device__ ElementList* getList()
        {
			trace(Element, "getList");
            return list;
        }

        __device__ Element* getNext();
        __device__ Element* getPrevious();

	#pragma endregion

	};

#endif
}}
