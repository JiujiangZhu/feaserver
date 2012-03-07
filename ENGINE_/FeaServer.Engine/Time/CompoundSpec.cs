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
    public class CompoundSpec<TEngine>
        where TEngine : IEngine
    {
        private static readonly Dictionary<uint, CompoundType> _typeIDs = new Dictionary<uint, CompoundType>();
        private static readonly Dictionary<CompoundType, CompoundSpec<TEngine>> _specs = new Dictionary<CompoundType, CompoundSpec<TEngine>>(new Dictionary<CompoundType, CompoundSpec<TEngine>>());
        public uint TypeID;
        public uint Length;
        public IElementType[] Types;

        private CompoundSpec(CompoundType compoundType)
        {
            TypeID = (uint)_typeIDs.Count; _typeIDs.Add(TypeID, compoundType);
            Types = compoundType.Types;
            Length = (uint)Types.Length;
        }

        public static CompoundType GetTypeByID(uint ID)
        {
            CompoundType type;
            if (_typeIDs.TryGetValue(ID, out type))
                return type;
            throw new ArgumentOutOfRangeException("ID");
        }

        public static CompoundSpec<TEngine> GetSpec(CompoundType compoundType)
        {
            CompoundSpec<TEngine> spec;
            if (!_specs.TryGetValue(compoundType, out spec))
                _specs.Add(compoundType, spec = new CompoundSpec<TEngine>(compoundType));
            return spec;
        }
    }
}
