//#pragma once
namespace Time { namespace Scheduler {
#ifdef _ELEMENTREF

#else
	#define ELEMENTREF

	class ElementRef : public System::LinkedListNode<ElementRef>
	{
	public:
		Element* Element;
		byte Metadata[MetadataSize];

	};

#endif
}}
