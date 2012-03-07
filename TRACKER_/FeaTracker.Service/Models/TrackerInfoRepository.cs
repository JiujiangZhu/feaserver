using System;
using System.Collections.Generic;
using System.ComponentModel.Composition;
namespace FeaTracker.Models
{
    /// <summary>
    /// ITrackerInfoRepository
    /// </summary>
    public interface ITrackerInfoRepository
    {
        IEnumerable<TrackerInfo> GetTrackers();
    }

    /// <summary>
    /// TrackerInfoRepository
    /// </summary>
    [Export(typeof(ITrackerInfoRepository))]
    public class TrackerInfoRepository : ITrackerInfoRepository
    {
        public IEnumerable<TrackerInfo> GetTrackers()
        {
            return new[] {
                new TrackerInfo { Time = 100 },
            };
        }
    }
}
