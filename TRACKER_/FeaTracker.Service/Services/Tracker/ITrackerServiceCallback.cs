using System;
using System.ServiceModel;
namespace FeaTracker.Services.Tracker
{
    /// <summary>
    /// ITrackerServiceCallback
    /// </summary>
    public interface ITrackerServiceCallback
    {
        [OperationContract(IsOneWay = true)]
        void OnMessage(string message);
    }
}
