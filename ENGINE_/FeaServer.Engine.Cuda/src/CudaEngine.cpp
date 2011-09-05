#pragma once
#include "CuContext.cpp"
#include "CudaElementTypeCollection.cpp"
using namespace System;
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	public ref class CudaEngine : IEngine
	{
	private:
		CuContext* _context;
		ElementTypeCollection^ _types;

	public:
		CudaEngine()
			: _context(new CuContext()), _types(gcnew CudaElementTypeCollection()) {
			if (!_context->Initialize())
				throw gcnew Exception(L"Unable to initalize CuContext");
		}
		~CudaEngine()
		{
			_context->Dispose();
			delete _context; _context = nullptr;
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
