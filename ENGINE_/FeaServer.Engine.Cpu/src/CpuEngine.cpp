#pragma once
#include "CpuElementTypeCollection.cpp"
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Threading;
namespace FeaServer { namespace Engine {
	public ref class CpuEngine : IEngine
	{
	private:
		ElementTypeCollection^ _types;
		FrugalThreadPool^ _threadPool;

	public:
		CpuEngine()
			: _types(gcnew CpuElementTypeCollection())
		{
			_threadPool = gcnew FrugalThreadPool(4, gcnew Action<Object^, Object^>(this, &CpuEngine::Executor), gcnew Func<Object^>(this, &CpuEngine::ThreadContextBuilder));
		}
		~CpuEngine()
		{
			delete _threadPool;
		}

#pragma region IEngine

		virtual ElementTable^ GetTable(Int32 shard)
		{
			Console::WriteLine("Cpu::GetTable");
			return nullptr;
		}

		virtual void LoadTable(ElementTable^ table, Int32 shard)
		{
			Console::WriteLine("Cpu::LoadTable");
		}

		virtual void UnloadTable(Int32 shard)
		{
			Console::WriteLine("Cpu::UnloadTable");
		}

		property ElementTypeCollection^ Types
		{
			virtual ElementTypeCollection^ get() { return _types; }
		}

		virtual void EvaluateFrame(UInt64 time)
		{
			Console::WriteLine("Cpu::EvaluateFrame");
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

