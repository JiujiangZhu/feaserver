#pragma once
namespace Time { namespace Scheduler {
#ifdef _ELEMENT

#else
	#define ELEMENT

	const static int MetadataSize = 4;
	class Element : public System::LinkedListNode<Element>
	{
	public:
		ElementScheduleStyle ScheduleStyle;
		byte Metadata[MetadataSize];

	};

#endif
}}
