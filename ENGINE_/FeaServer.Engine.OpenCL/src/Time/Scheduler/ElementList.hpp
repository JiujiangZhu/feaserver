#pragma once
#include "Element.hpp"
namespace Time { namespace Scheduler {
#ifdef _ELEMENT

#else
	#define ELEMENTLIST
	class ElementList : public System::LinkedList<Element>
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
		
	};
#endif
}}