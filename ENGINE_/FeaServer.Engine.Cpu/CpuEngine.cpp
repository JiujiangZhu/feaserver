#pragma once
#include "Engine.h"
using namespace System;
using namespace System::Collections::Generic;
namespace FeaServer { namespace Engine {
	public ref class CpuEngine : IEngine
	{
	private:
		ObjectCollection^ _objects;
		TypeCollection^ _types;
		FrugalThreadPool^ _threadPool;

	public:
		CpuEngine()
			: _objects(gcnew ObjectCollection(gcnew List<IObject^>()))
			, _types(gcnew TypeCollection(gcnew List<IType^>())) {
			_threadPool = gcnew FrugalThreadPool(4, gcnew Action<Object^, Object^>(this, &CpuEngineProvider::Executor), gcnew Func<Object^>(this, &CpuEngineProvider::ThreadContextBuilder));
		}
		~CpuEngine()
		{
			delete _threadPool;
		}

#pragma region Engine
		property ObjectCollection^ Objects
		{
			virtual ObjectCollection^ get() { return _objects; }
		}

		property TypeCollection^ Types
		{
			virtual TypeCollection^ get() { return _types; }
		}

		virtual void EvaluateFrame(Int64 time)
		{
		}
#pragma endregion

	private:
		void Executor(Object^ obj, Object^ threadContext)
		{
			return;
		}
		Object^ ThreadContextBuilder()
		{
			return nullptr;
		}
	};
}}

