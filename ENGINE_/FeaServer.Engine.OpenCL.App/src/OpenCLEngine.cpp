#pragma once
#include "OpenCLElementTypeCollection.h"
using namespace System;
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	public ref class OpenCLEngine : IEngine
	{
	private:
		ElementTypeCollection^ _types;

	public:
		OpenCLEngine()
			: _types(gcnew OpenCLElementTypeCollection()) {
		}
		~OpenCLEngine()
		{
		}

#pragma region IEngine

		virtual ElementTable^ GetTable(Int32 shard)
		{
			return nullptr;
		}

		virtual void LoadTable(ElementTable^ table, Int32 shard)
		{
		}

		virtual void UnloadTable(Int32 shard)
		{
		}

		property ElementTypeCollection^ Types
		{
			virtual ElementTypeCollection^ get() { return _types; }
		}

		virtual void EvaluateFrame(UInt64 time)
		{
			//s_engine.EvaluateFrame(time);
		}

#pragma endregion

	};
}}
