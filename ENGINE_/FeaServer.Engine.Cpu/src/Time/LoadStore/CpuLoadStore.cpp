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
#include "..\..\Core.h"
#include "..\Element.hpp"
#include "..\..\CpuEngine.cpp"
using namespace System;
using namespace System::IO;
using namespace System::Collections::Generic;

namespace FeaServer { namespace Engine { namespace Time {
	#define CPULOADSTORE_MAGIC (unsigned short)0x3412
	public ref class CpuLoadStore
	{
	private:
		LoadStoreContext<CpuEngine^>^ _ctx;

	public:
		static CpuLoadStore()
		{
			ElementSpec<CpuEngine^>::SizeOfElement = sizeof(::Time::Element);
		}
		CpuLoadStore()
			: _ctx(gcnew LoadStoreContext<CpuEngine^>(nullptr, nullptr, nullptr))
		{
		}

	//private:
		//size_t Init(Compound% compound, BinaryWriter^ w)
		//{
		//	w->Write(CPULOADSTORE_MAGIC);
		//	CompoundSpec^ spec = CompoundSpec::GetSpec(compound.Type);
		//	int length = spec->Length;
		//	w->Write(length);
		//	size_t size = array_getSize(void*, length);
		//	//
		//	int n2 = 0;
		//	int dataSize = 0;
		//	array<char, 1>^ data;
		//	array<int>^ typesSizeInBytes = spec->TypesSizeInBytes;
		//	for (int index = 0; index < spec->Length; index++)
		//	{
		//		int pitch = typesSizeInBytes[index];
		//		size += array_getSizeEx(pitch, n2);
		//		w->Write(pitch);
		//		w->Write(n2);
		//		// add data2 to dataStream
		//		BinaryWriter^ specW = GetSpecWriter(spec);
		//		specW->Write(dataSize);
		//		specW->Write(data, 0, dataSize);
		//	}
		//	return size;
		//}
	};
}}}
