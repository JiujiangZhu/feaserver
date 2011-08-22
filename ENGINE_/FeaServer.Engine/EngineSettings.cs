namespace FeaServer.Engine
{
    /// <summary>
    /// EngineSettings
    /// </summary>
    public static class EngineSettings
    {
        public const int MaxTimeslices = 1000;
        public const int MaxHibernateSegments = 3;
        public const int MaxWorkingFractions = 10;
        public const ulong MaxTimeslicesTime = (MaxTimeslices << TimePrecision.TimePrecisionBits);
    }
}
