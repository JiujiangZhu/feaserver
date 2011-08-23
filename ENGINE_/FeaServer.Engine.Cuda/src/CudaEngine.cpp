#pragma once
#include "CuContext.h"
#include "CudaElementTypeCollection.h"
//#include "Engine.h"
using namespace System;
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	public ref class CudaEngine : IEngine
	{
	private:
		CuContext* _context;
		ElementTypeCollection^ _elementTypes;

	public:
		CudaEngine()
			: _context(new CuContext()), _elementTypes(gcnew CudaElementTypeCollection()) {
			if (!_context->Initialize())
				throw gcnew Exception(L"Unable to initalize CuContext");
		}
		~CudaEngine()
		{
			_context->Dispose();
			delete _context; _context = nullptr;
		}

#pragma region IEngine

		virtual IEnumerable<IElement^>^ GetElements(Int32 shard)
		{
			return nullptr;
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
