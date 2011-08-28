#pragma once
#include "CpuElementTypeCollection.h"
//#include "Engine.h"
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Threading;
namespace FeaServer { namespace Engine {
	public ref class CpuEngine : IEngine
	{
	private:
		ElementTypeCollection^ _elementTypes;
		FrugalThreadPool^ _threadPool;

	public:
		CpuEngine()
			: _elementTypes(gcnew CpuElementTypeCollection()) {
			_threadPool = gcnew FrugalThreadPool(4, gcnew Action<Object^, Object^>(this, &CpuEngine::Executor), gcnew Func<Object^>(this, &CpuEngine::ThreadContextBuilder));
		}
		~CpuEngine()
		{
			delete _threadPool;
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
		}

#pragma endregion

	private:
		void Executor(Object^ obj, Object^ threadContext)
		{
		}

		Object^ ThreadContextBuilder()
		{
			return nullptr;
		}
	};
}}

