#pragma once
using namespace System;
namespace FeaServer { namespace Engine {
	private ref class CpuElementTypeCollection : ElementTypeCollection
	{
	public:
		virtual void InsertItem(Int32 index, IElementType^ item) override
		{
			Console::WriteLine("Cpu::Insert");
		}

		virtual void SetItem(Int32 index, IElementType^ item) override
		{
			throw gcnew NotSupportedException();
		}
	};
}}
