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
