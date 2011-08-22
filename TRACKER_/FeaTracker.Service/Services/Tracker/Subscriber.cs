using System;
using System.ServiceModel;
using System.Threading;
namespace FeaTracker.Services.Tracker
{
    /// <summary>
    /// Subscriber
    /// </summary>
    public class Subscriber
    {
        internal static long VersionID = long.MaxValue;

        public Subscriber(OperationContext context, Subscription subscription)
        {
            var lastColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("Add:  {0}", context.SessionId);
            Console.ForegroundColor = lastColor;
            //
            Subscription = subscription;
            Interlocked.Increment(ref VersionID);
        }

        public Subscription Subscription { get; private set; }

        public void Remove(OperationContext context)
        {
            var lastColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("Del:  {0}", (context != null ? context.SessionId : "forced"));
            Console.ForegroundColor = lastColor;
            Interlocked.Increment(ref VersionID);
        }

        public void Welcome(ITrackerServiceCallback callback)
        {
            callback.OnMessage("Welcome");
        }
    }
}
