using System;
using System.Linq;
using System.ServiceModel;
using System.Collections.Generic;
using FeaTracker.Services.Tracker;
namespace FeaTracker.Services
{
    /// <summary>
    /// ITrackerService
    /// </summary>
    [ServiceContract(CallbackContract = typeof(ITrackerServiceCallback))]
    public interface ITrackerService
    {
        #region Subscribe
        [OperationContract]
        bool Subscribe(Subscription subscription);

        [OperationContract]
        bool Unsubscribe();
        #endregion
    }

    /// <summary>
    /// TrackerService
    /// </summary>
    public class TrackerService : ITrackerService
    {
        private static readonly object _lock = new object();

        public static void Broadcast(Action<ITrackerServiceCallback> action, Predicate<Subscriber> predicate)
        {
            var versionID = Subscriber.VersionID;
            var removes = new List<ITrackerServiceCallback>();
            lock (_lock)
            {
                foreach (var subscriber in _subscribers)
                {
                    var key = subscriber.Key;
                    var value = subscriber.Value;
                    if (((ICommunicationObject)key).State == CommunicationState.Opened)
                    {
                        if ((predicate == null) || predicate(value))
                            action(key);
                    }
                    else
                    {
                        if (removes == null)
                            removes = new List<ITrackerServiceCallback>();
                        removes.Add(key);
                        value.Remove(null);
                    }
                }
                if (removes != null)
                    foreach (var remove in removes)
                        _subscribers.Remove(remove);
            }
        }

        #region Subscribe

        private static readonly Dictionary<ITrackerServiceCallback, Subscriber> _subscribers = new Dictionary<ITrackerServiceCallback, Subscriber>();

        public bool Subscribe(Subscription subscription)
        {
            //try
            //{
            var context = OperationContext.Current;
            var callback = context.GetCallbackChannel<ITrackerServiceCallback>();
            if (_subscribers.ContainsKey(callback))
                return true;
            var subscriber = new Subscriber(context, subscription);
            subscriber.Welcome(callback);
            lock (_lock)
                _subscribers.Add(callback, subscriber);
            return true;
            //}
            //catch { return false; }
        }

        public bool Unsubscribe()
        {
            //try
            //{
            var context = OperationContext.Current;
            var callback = context.GetCallbackChannel<ITrackerServiceCallback>();
            Subscriber subscriber;
            if (!_subscribers.TryGetValue(callback, out subscriber))
                return true;
            lock (_lock)
                _subscribers.Remove(callback);
            subscriber.Remove(context);
            return true;
            //}
            //catch { return false; }
        }

        #endregion
    }
}
