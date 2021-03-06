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
#include "ElementSpec.cpp"
using namespace System;
using namespace System::Collections::Generic;

namespace FeaServer { namespace Engine { namespace Time2 {
	public ref struct CompoundSpec
	{
	private:
		static Dictionary<CompoundType^, CompoundSpec^> _specs = gcnew Dictionary<CompoundType^, CompoundSpec^>();

	public:
		CompoundSpec(CompoundType^ compoundType)
		{
			Types = compoundType->Types;
			Length = Types->Length;
			TypesSizeInBytes = gcnew array<int>(Length);
			for (int index = 0; index < Length; index++)
			{
				ElementSpec^ elementSpec = ElementSpec::GetSpec(Types[index]);
				TypesSizeInBytes[index] = elementSpec->SizeInBytes;
			}
		}

		static CompoundSpec^ GetSpec(CompoundType^ compoundType)
		{
			CompoundSpec^ spec;
			if (!_specs.TryGetValue(compoundType, spec))
			{
				spec = gcnew CompoundSpec(compoundType);
				_specs.Add(compoundType, spec);
			}
			return spec;
		}

	public:
		int Length;
		array<IElementType^>^ Types;
		array<int>^ TypesSizeInBytes;
	};

}}}

