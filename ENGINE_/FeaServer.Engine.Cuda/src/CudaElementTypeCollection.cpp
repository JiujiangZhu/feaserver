#pragma once
using namespace System;
namespace FeaServer { namespace Engine {
	private ref class CudaElementTypeCollection : ElementTypeCollection
	{
	public:
		virtual void InsertItem(Int32 index, IElementType^ item) override
		{
			Console::WriteLine("Cuda::Insert");
		}

		virtual void SetItem(Int32 index, IElementType^ item) override
		{
			throw gcnew NotSupportedException();
		}
	};
}}
