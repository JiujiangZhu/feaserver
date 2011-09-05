#pragma once
#include "LinkKind.h"
#include "ListKind.h"
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	struct SliceNode
	{
	public:
		/*SliceNode(LinkKind* value)
			: LinkKind(value) { }*/
		ListKind::List* List;
		LinkKind* Chain;
	};
}}
