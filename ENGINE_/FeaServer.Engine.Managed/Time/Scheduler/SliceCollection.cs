using System;
using System.Collections.Generic;
namespace FeaServer.Engine.Time.Scheduler
{
    internal class SliceCollection
    {
        private ulong _currentSlice = 0;
        private Slice[] _slices = new Slice[EngineSettings.MaxTimeslices];
        private HibernateCollection _hibernates;
        private SliceFractionCache _fractionCache = new SliceFractionCache();

        public SliceCollection()
        {
            for (int sliceIndex = 0; sliceIndex < _slices.Length; sliceIndex++)
                _slices[sliceIndex].xtor();
            _hibernates.xtor();
        }

        public void Schedule(Element element, ulong time)
        {
            Console.WriteLine("Timeline: Add %d", TimePrec.DecodeTime(time));
            unchecked
            {
                var slice = (ulong)(time >> TimePrec.TimePrecisionBits);
                var fraction = (ulong)(time & TimePrec.TimePrecisionMask);
                if (slice < EngineSettings.MaxTimeslices)
                {
                    // first fraction
                    if (slice == 0)
                        _fractionCache.EnsureCache(fraction);
                    // roll timeslice for index
                    slice += _currentSlice;
                    if (slice >= EngineSettings.MaxTimeslices)
                        slice -= EngineSettings.MaxTimeslices;
                    _slices[slice].Fractions.Schedule(element, fraction);
                }
                else
                    _hibernates.Hibernate(element, time);
            }
        }

        public void ScheduleRange(IEnumerable<Tuple<Element, ulong>> elements)
        {
            foreach (var element in elements)
                Schedule(element.Item1, element.Item2);
        }

        private void MoveNextSlice()
        {
            _currentSlice++;
            if (_currentSlice >= EngineSettings.MaxTimeslices)
            {
                _currentSlice = 0;
                _hibernates.DeHibernate(this);
            }
        }

        #region Evaluate

        public bool EvaluateFrame(ulong frameTime, Action<SliceNode> evaluateNode)
        {
            Console.WriteLine("Timeline: EvaluateFrame %d", TimePrec.DecodeTime(frameTime));
            unchecked
            {
                bool firstLoop = true;
                _fractionCache.BeginFrame();
                long timeRemaining = (long)frameTime;
                while (timeRemaining <= 0)
                {
                    if (!_fractionCache.BeginSlice(_slices[_currentSlice]))
                        // no fractions available, advance a wholeTime
                        timeRemaining -= (long)TimePrec.TimeScaler;
                    else
                    {
                        // first-time time adjust
                        if (firstLoop)
                        {
                            firstLoop = false;
                            //time += (long)_fractionCache.CurrentFraction;
                        }
                        long elapsedTime;
                        if (!EvaluateSlice(timeRemaining, out elapsedTime, evaluateNode))
                        {
                            // slice not completely evaluated, continue with same slice next frame
                            _fractionCache.EndFrame();
                            return false;
                        }
                        // advance an elapsedTime
                        timeRemaining -= elapsedTime;
                    }
                    _fractionCache.EndSlice();
                    // advance a slice
                    MoveNextSlice();
                }
                _fractionCache.EndFrame();
                return true;
            }
        }

        private bool EvaluateSlice(long time, out long elapsedTime, Action<SliceNode> evaluateNode)
        {
            //ulong lastFraction;
            ulong currentFraction;
            while ((currentFraction = _fractionCache.BeginSliceFraction()) < ulong.MaxValue)
            {
                //// advance time-slice & check for escape
                //time -= (long)(currentFraction - lastFraction);
                //if (time <= 0)
                //{
                //    elapsedTime = (long)(TimePrecision.TimeScaler - lastFraction);
                //    return false;
                //}
                evaluateNode(_fractionCache.MoveNextSliceFraction());
                _fractionCache.EndSliceFraction();
            }
            // advance time
            //_isRebuildWorkingFractions = true;
            elapsedTime = 0; // (long)(TimePrecision.TimeScaler - lastFraction);
            return true;
        }

        #endregion
    }
}
