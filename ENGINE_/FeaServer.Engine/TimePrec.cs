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