#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
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
			//_threadPool = gcnew FrugalThreadPool(4, gcnew Action<Object^, Object^>(this, &CpuEngine::Executor), gcnew Func<Object^>(this, &CpuEngine::ThreadContextBuilder));
		}
		~CpuEngine()
		{
			delete _threadPool;
		}

#pragma region IEngine

		virtual CompoundTable^ GetTable(Int32 shard)
		{
			Console::WriteLine("Cpu::GetTable");
			return nullptr;
		}

		virtual void LoadTable(CompoundTable^ table, Int32 shard)
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

