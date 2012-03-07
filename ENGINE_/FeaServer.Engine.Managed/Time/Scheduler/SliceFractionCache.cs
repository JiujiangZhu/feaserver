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

namespace FeaServer.Engine.Time.Scheduler
{
    internal class SliceFractionCache
    {
        //: (sorted)dictionary removed are expensive. virtually remove with a window (_minWorkingFraction).
        //private static readonly ReverseComparer _reverseComparer = new ReverseComparer();
        private SliceFractionCollection _sliceFractions;
        private ulong[] _fractions = new ulong[EngineSettings.MaxWorkingFractions];
        private int _currentFractionIndex;
        private ulong _minFraction;
        private ulong _maxFraction;
        private ulong _currentFraction;

        //public class ReverseComparer : IComparer<ulong>
        //{
        //    public int Compare(ulong x, ulong y) { return (x < y ? 1 : (x > y ? -1 : 0)); }
        //}

        public ulong CurrentFraction
        {
            get { return _currentFraction; }
        }

        public bool RequiresRebuild;

        public void xtor()
        {
            Console.WriteLine("SliceFractionCache:xtor");
        }
        public void Dispose()
        {
            Console.WriteLine("SliceFractionCache:Dispose");
        }

        public void EnsureCache(ulong fraction)
        {
            if (fraction < _maxFraction)
                RequiresRebuild = true;
        }

        private void Rebuild()
        {
            //if ((_sliceFractions == null) || (_sliceFractions.Count == 0))
            //    throw new NullReferenceException();
            //var keys = _sliceFractions.Keys;
            //var values = new ulong[keys.Count];
            //keys.CopyTo(values, 0);
            //Array.Sort(values, _reverseComparer);
            //int fractionsLength = values.Length;
            //_currentFractionIndex = Math.Min(fractionsLength, EngineSettings.MaxTimeslices);
            //Array.Copy(values, fractionsLength - _currentFractionIndex, _fractions, 0, _currentFractionIndex);
            //_maxFraction = _fractions[0];
            //_currentFraction = _fractions[--_currentFractionIndex];
            //RequiresRebuild = false;
        }

        public SliceFraction MoveNextSliceFraction()
        {
            _minFraction = _currentFraction; //: fractions.Remove(fractionTime);
            _currentFraction = (_currentFractionIndex > 0 ? _fractions[--_currentFractionIndex] : ulong.MaxValue); //: repnz requires one less register
            return _sliceFractions[_minFraction];
        }

        #region Evaluate

        internal void BeginFrame()
        {
            RequiresRebuild = true;
        }

        internal bool BeginSlice(Slice slice)
        {
            _sliceFractions = slice.Fractions;
            if (_sliceFractions.Count == 0)
            {
                RequiresRebuild = true;
                return false;
            }
            Rebuild();
            return true;
        }

        public ulong BeginSliceFraction()
        {
            return CurrentFraction;
        }

        public void EndSliceFraction()
        {
            if ((RequiresRebuild) || (_currentFraction == _maxFraction))
                Rebuild();
        }

        internal void EndSlice()
        {
        }

        internal void EndFrame()
        {
            RequiresRebuild = true;
        }

        #endregion
    }
}
