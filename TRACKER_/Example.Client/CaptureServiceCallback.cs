using System;
using Example.Client.Tracker;
using System.ServiceModel;
namespace Example
{
    [CallbackBehavior(ConcurrencyMode = ConcurrencyMode.Reentrant)]
    public class TrackerServiceCallback : ITrackerServiceCallback, IDisposable
    {
        private readonly TrackerServiceClient _client;

        public TrackerServiceCallback()
        {
            _client = new TrackerServiceClient(new InstanceContext(this));
            _client.Subscribe(new Subscription { });
        }
        public void Dispose()
        {
            _client.Unsubscribe();
            _client.Close();
        }

        public void OnMessage(string message)
        {
            Console.WriteLine(message);
        }
    }
}
