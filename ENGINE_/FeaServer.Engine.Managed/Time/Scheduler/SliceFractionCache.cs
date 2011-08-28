using System;
using System.Collections.Generic;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class SliceFractionCache
    {
        //: (sorted)dictionary removed are expensive. virtually remove with a window (_minWorkingFraction).
        private static readonly ReverseComparer _reverseComparer = new ReverseComparer();
        private SliceFractionCollection _sliceFractions;
        private ulong[] _fractions = new ulong[EngineSettings.MaxWorkingFractions];
        private int _currentFractionIndex;
        private ulong _minFraction;
        private ulong _maxFraction;
        private ulong _currentFraction;

        public class ReverseComparer : IComparer<ulong>
        {
            public int Compare(ulong x, ulong y) { return (x < y ? 1 : (x > y ? -1 : 0)); }
        }

        public ulong CurrentFraction
        {
            get { return _currentFraction; }
        }

        public bool RequiresRebuild { get; private set; }

        private void Rebuild()
        {
            if ((_sliceFractions == null) || (_sliceFractions.Count == 0))
                throw new NullReferenceException();
            var keys = _sliceFractions.Keys;
            var values = new ulong[keys.Count];
            keys.CopyTo(values, 0);
            Array.Sort(values, _reverseComparer);
            int fractionsLength = values.Length;
            _currentFractionIndex = Math.Min(fractionsLength, EngineSettings.MaxTimeslices);
            Array.Copy(values, fractionsLength - _currentFractionIndex, _fractions, 0, _currentFractionIndex);
            _maxFraction = _fractions[0];
            _currentFraction = _fractions[--_currentFractionIndex];
            RequiresRebuild = false;
        }

        public SliceNode MoveNextSliceFraction()
        {
            _minFraction = _currentFraction; //: fractions.Remove(fractionTime);
            _currentFraction = (_currentFractionIndex > 0 ? _fractions[--_currentFractionIndex] : ulong.MaxValue); //: repnz requires one less register
            return _sliceFractions[_minFraction];
        }

        public void EnsureCache(ulong fraction)
        {
            if (fraction < _maxFraction)
                RequiresRebuild = true;
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
