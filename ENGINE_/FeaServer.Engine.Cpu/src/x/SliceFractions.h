#pragma once
#include "SliceNode.h"
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	struct SliceFractions //: SortedDictionary<unsigned long, Slice>
	{
	public:
		bool TryGetValue(unsigned long fraction, SliceNode* node)
		{
			return false;
		}

		void Add(unsigned long fraction, LinkKind* value)
		{
		}
		void Add(unsigned long fraction, ListKind* value)
		{
		}
	};
}}
