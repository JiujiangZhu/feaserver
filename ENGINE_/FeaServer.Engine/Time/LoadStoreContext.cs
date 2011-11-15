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
using System.IO;
using System.Linq;
namespace FeaServer.Engine.Time
{
    public interface ILoadStoreContext
    {
    }

    public class LoadStoreContext<TEngine> : ILoadStoreContext
        where TEngine : IEngine
    {
        public const ushort LOADSTORE_MAGIC = 0x3412;
        public const byte LOADSTORE_BLOCKTYPE = 0x01;
        public const byte LOADSTORE_HEADTYPE = 0x00;
        public static int SizeOfSize_t;
        private Dictionary<CompoundSpec<TEngine>, SpecWriter> _specWriters = new Dictionary<CompoundSpec<TEngine>, SpecWriter>();
        private List<Tuple<ulong, long>> _mallocs = new List<Tuple<ulong, long>>();
        private List<long> _adjustSIndexs = new List<long>();
        private MemoryStream _s = new MemoryStream();
        private BinaryWriter _w;
        private BinaryReader _r;
        private Func<long, ulong> _xAlloc;
        private Action<MemoryStream, FileAccess, ulong, long> _xTransfer;
        private Action<ulong> _xFree;

        private struct SpecWriter
        {
            public BinaryWriter W;
            public ulong LastEveryAddress;
        }

        public LoadStoreContext(Func<long, ulong> alloc, Action<MemoryStream, FileAccess, ulong, long> transfer, Action<ulong> free)
        {
            if (alloc == null)
                throw new ArgumentNullException("alloc");
            if (transfer == null)
                throw new ArgumentNullException("transfer");
            if (free == null)
                throw new ArgumentNullException("free");
            _xAlloc = alloc;
            _xTransfer = transfer;
            _xFree = free;
            _w = new BinaryWriter(_s);
            _r = new BinaryReader(_s);
        }
        public void Dispose()
        {
            foreach (var malloc in _mallocs)
                _xFree(malloc.Item1);
            _mallocs.Clear();
        }

        private SpecWriter GetSpecWriter(CompoundSpec<TEngine> spec)
        {
            SpecWriter w;
            if (!_specWriters.TryGetValue(spec, out w))
            {
                w = new SpecWriter
                {
                    W = new BinaryWriter(new MemoryStream()),
                };
                _specWriters.Add(spec, w);
            }
            return w;
        }

        public IEnumerable<Compound> Store()
        {
            foreach (var malloc in _mallocs)
            {
                var items = StoreGroup(malloc.Item1, malloc.Item2);
                if (items != null)
                    foreach (var item in items)
                        yield return item;
            }
        }

        public void Load(IEnumerable<Compound> compounds)
        {
            var items = new List<Compound>();
            foreach (var compound in compounds)
            {
                items.Add(compound);
                if (items.Count == 16)
                {
                    LoadGroup(items);
                    items.Clear();
                }
            }
            if (items.Count > 0)
                LoadGroup(items);
            LoadHeader();
        }

        private IEnumerable<Compound> StoreGroup(ulong address, long length)
        {
            _xTransfer(_s, FileAccess.Read, address, length);
            if (_r.ReadUInt16() != LOADSTORE_MAGIC)
                throw new Exception("Bad Magic");
            if (_r.ReadByte() != LOADSTORE_BLOCKTYPE)
                return null;
            var items = new List<Compound>();
            for (int index = 0; index < _r.ReadUInt16(); index++)
                items.Add(FromStream());
            _s.SetLength(0);
            return items;
        }

        private void LoadGroup(IList<Compound> compounds)
        {
            _w.Write(LOADSTORE_MAGIC);
            _w.Write(LOADSTORE_BLOCKTYPE);
            _w.Write((ushort)compounds.Count);
            foreach (var compound in compounds)
                ToStream(compound);
            var length = _s.Length;
            var address = _xAlloc(length);
            _mallocs.Add(new Tuple<ulong, long>(address, length));
            Adjust(address);
            _xTransfer(_s, FileAccess.Write, address, length);
            _s.SetLength(0);
        }

        private void LoadHeader()
        {
            _w.Write(LOADSTORE_MAGIC);
            _w.Write(LOADSTORE_HEADTYPE);
            _s.SetLength(0);
        }

        private Compound FromStream()
        {
            var compoundID = _r.ReadUInt32();
            // write compoundType
            var compoundType = CompoundSpec<TEngine>.GetTypeByID(_r.ReadUInt32());
            var spec = CompoundSpec<TEngine>.GetSpec(compoundType);
            if (_r.ReadUInt16() != spec.Length)
                throw new Exception();
            _s.Seek(spec.Length * sizeof(long), SeekOrigin.Current);
            var compoundElements = new CompoundItem[spec.Length][];
            for (int specIndex = 0; specIndex < spec.Length; specIndex++)
            {
                var elementSpec = ElementSpec<TEngine>.GetSpec(spec.Types[specIndex]);
                var sizeOfElement = ElementSpec<TEngine>.SizeOfElement;
                var stateSizeInBytes = elementSpec.StateSizeInBytes;
                if (_r.ReadUInt32() != elementSpec.TotalSizeInBytes)
                    throw new Exception();
                var elementLength = _r.ReadUInt32();
                var elements = new CompoundItem[elementLength];
                for (int elementIndex = 0; elementIndex < elementLength; elementIndex++)
                {
                    elements[elementIndex].ID = _r.ReadUInt32();
                    _s.Seek(sizeOfElement, SeekOrigin.Current);
                    var remainingPitch = (long)stateSizeInBytes;
                    var dataSizeInBytes = elementSpec.DataSizeInBytes;
                    if (dataSizeInBytes > 0)
                    {
                        elements[elementIndex].Data = _r.ReadBytes(dataSizeInBytes);
                        remainingPitch -= dataSizeInBytes;
                    }
                    if (remainingPitch < 0)
                        throw new OverflowException("Data larger then StateSize");
                    if (remainingPitch > 0)
                        _s.Seek(remainingPitch, SeekOrigin.Current);
                }
                compoundElements[specIndex] = elements;
            }
            return new Compound
            {
                ID = compoundID,
                Elements = compoundElements,
                Type = compoundType,
            };
        }

        private void ToStream(Compound compound)
        {
            _w.Write(compound.ID);
            // write compoundType
            var spec = CompoundSpec<TEngine>.GetSpec(compound.Type);
            _w.Write(spec.ID);
            _w.Write(spec.Length); // #arrayLength
            var specSIndexs = new long[spec.Length];
            var specArraySIndex = _s.Position;
            _s.Seek(spec.Length * sizeof(ulong), SeekOrigin.Current);
            for (int specIndex = 0; specIndex < spec.Length; specIndex++)
            {
                var elementSpec = ElementSpec<TEngine>.GetSpec(spec.Types[specIndex]);
                var sizeOfElement = ElementSpec<TEngine>.SizeOfElement;
                var stateSizeInBytes = elementSpec.StateSizeInBytes;
                //
                var elements = compound.Elements[specIndex];
                _w.Write(elementSpec.TotalSizeInBytes); // #arrayPitch
                _w.Write((uint)elements.Length); // #arrayLength
                specSIndexs[specIndex] = _s.Position;
                foreach (var element in elements)
                {
                    _w.Write(element.ID);
                    _s.Seek(sizeOfElement, SeekOrigin.Current);
                    var remainingPitch = (long)stateSizeInBytes;
                    var data = element.Data;
                    if (data != null)
                    {
                        _w.Write(data);
                        remainingPitch -= data.Length;
                    }
                    if (remainingPitch < 0)
                        throw new OverflowException("Data larger then StateSize");
                    if (remainingPitch > 0)
                        _s.Seek(remainingPitch, SeekOrigin.Current);
                }
            }
            // replace specSIndex
            var lastSIndex = _s.Position;
            _s.Seek(specArraySIndex, SeekOrigin.Begin);
            foreach (var specSIndex in specSIndexs)
            {
                _adjustSIndexs.Add(_s.Position);
                _w.Write((ulong)specSIndex);
            }
            _s.Position = lastSIndex;
        }

        private void Adjust(ulong address)
        {
            // replace specSIndex
            var lastSIndex = _s.Position;
            foreach (var adjustSIndex in _adjustSIndexs)
            {
                _s.Position = adjustSIndex;
                //_s.Write((long)specSIndex);
            }
            _s.Position = lastSIndex;
        }

        //private static ulong array_getSize(int res, int length) { return (ulong)((res * length) + SizeOfSize_t); }
    }
}
