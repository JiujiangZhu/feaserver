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
    internal class SliceCollection
    {
        private ulong _currentSlice;
        private byte _currentHibernate;
        private Slice[] _slices = new Slice[EngineSettings.MaxTimeslices];
        private HibernateCollection _hibernates;
        private SliceFractionCache _fractionCache = new SliceFractionCache();

        public SliceCollection()
        {
            Console.WriteLine("SliceCollection:ctor");
            _currentSlice = 0;
            _currentHibernate = 0;
            for (int sliceIndex = 0; sliceIndex < _slices.Length; sliceIndex++)
                _slices[sliceIndex].xtor();
            _hibernates.xtor();
            _fractionCache.xtor();
        }
        public void Dispose()
        {
            Console.WriteLine("SliceCollection:Dispose");
            for (int sliceIndex = 0; sliceIndex < _slices.Length; sliceIndex++)
                _slices[sliceIndex].Dispose();
            _hibernates.Dispose();
            _fractionCache.Dispose();
        }

        public void Schedule(Element element, ulong time)
        {
            Console.WriteLine("SliceCollection:Schedule {0}", TimePrec.DecodeTime(time));
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

        public void MoveNextSlice()
        {
            Console.WriteLine("SliceCollection:MoveNextSlice {0}", _currentSlice);
            _slices[_currentSlice].Dispose();
            if (++_currentSlice >= EngineSettings.MaxTimeslices)
            {
                _currentSlice = 0;
                _hibernates.DeHibernate(this);
                if (++_currentHibernate >= EngineSettings.HibernatesTillReShuffle)
                {
                    _currentHibernate = 0;
                    _hibernates.ReShuffle();
                }
            }
        }

        #region Evaluate

        public void EvaluateFrameBegin(ulong frameTime)
        {
            Console.WriteLine("SliceCollection:EvaluateFrameBegin {0}", TimePrec.DecodeTime(frameTime));
        }

        public void EvaluateFrameEnd()
        {
            Console.WriteLine("SliceCollection:EvaluateFrameEnd");
        }



        public bool EvaluateFrame(ulong frameTime, Action<SliceFraction> evaluateNode)
        {
            Console.WriteLine("SliceCollection:EvaluateFrame {0}", TimePrec.DecodeTime(frameTime));
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

        private bool EvaluateSlice(long time, out long elapsedTime, Action<SliceFraction> evaluateNode)
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
