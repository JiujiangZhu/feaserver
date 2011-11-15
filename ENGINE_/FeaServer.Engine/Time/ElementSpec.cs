#region License
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
#endregion
using System;
using System.Collections.Generic;

namespace FeaServer.Engine.Time
{
    public class ElementSpec<TEngine>
        where TEngine : IEngine
    {
        private static readonly Dictionary<IElementType, ElementSpec<TEngine>> _specs = new Dictionary<IElementType, ElementSpec<TEngine>>(new Dictionary<IElementType, ElementSpec<TEngine>>());
        public static uint SizeOfElement;
        public uint TotalSizeInBytes;
        public uint StateSizeInBytes;
        public int DataSizeInBytes;

        private ElementSpec(IElementType elementType)
        {
            var image = elementType.GetImage(Foo(typeof(TEngine)));
            TotalSizeInBytes = image.StateSizeInBytes + SizeOfElement + sizeof(uint);
            StateSizeInBytes = image.StateSizeInBytes;
            DataSizeInBytes = image.DataSizeInBytes;
        }

        public static ElementSpec<TEngine> GetSpec(IElementType elementType)
        {
            ElementSpec<TEngine> spec;
            if (!_specs.TryGetValue(elementType, out spec))
            {
                spec = new ElementSpec<TEngine>(elementType);
                _specs.Add(elementType, spec);
            }
            return spec;
        }

        private static EngineProvider Foo(Type type)
        {
            return EngineProvider.Cpu;
        }
    }
}
