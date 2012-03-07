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

namespace FeaServer.Engine
{
    /// <summary>
    /// TimePrec
    /// </summary>
    public static class TimePrec
    {
        public const int TimePrecisionBits = 4;
        public const ulong TimePrecisionMask = (1 << TimePrecisionBits) - 1;
        public const ulong TimeScaler = (1 << TimePrec.TimePrecisionBits);
        private const decimal TimeScaleUnit = (1M / TimeScaler);

        /// <summary>
        /// Parses the time.
        /// </summary>
        /// <param name="time">The time.</param>
        /// <returns></returns>
        public static ulong EncodeTime(this decimal time)
        {
            var integer = (int)Math.Truncate(time);
            var fraction = (decimal)time - integer;
            return ((ulong)(integer << TimePrecisionBits) + (ulong)Math.Round(fraction / TimeScaleUnit));
        }

        /// <summary>
        /// Formats the time.
        /// </summary>
        /// <param name="time">The time.</param>
        /// <returns></returns>
        public static decimal DecodeTime(this ulong time)
        {
            var integer = (ulong)(time >> TimePrec.TimePrecisionBits);
            var fraction = (ulong)(time & TimePrec.TimePrecisionMask);
            return integer + (fraction * TimeScaleUnit);
        }
    }
}
