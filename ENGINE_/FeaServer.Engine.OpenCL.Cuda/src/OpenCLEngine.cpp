#pragma once
#include "OpenCLElementTypeCollection.h"
using namespace System;
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	public ref class OpenCLEngine : IEngine
	{
	private:
		ElementTypeCollection^ _elementTypes;

	public:
		OpenCLEngine()
			: _elementTypes(gcnew OpenCLElementTypeCollection()) {
		}
		~OpenCLEngine()
		{
		}

#pragma region IEngine

		virtual IEnumerable<IElement^>^ GetElements(Int32 shard)
		{
			return nullptr;
		}

		virtual void LoadElements(IEnumerable<IElement^>^ elements, Int32 shard)
		{
		}

		property ElementTypeCollection^ ElementTypes
		{
			virtual ElementTypeCollection^ get() { return _elementTypes; }
		}

		virtual void EvaluateFrame(UInt64 time)
		{
			//s_engine.EvaluateFrame(time);
		}

#pragma endregion

	};
}}
