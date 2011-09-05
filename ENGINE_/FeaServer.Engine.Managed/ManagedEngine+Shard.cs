using System;
using FeaServer.Engine.Time.Scheduler;
using System.Threading;
namespace FeaServer.Engine
{
    public partial class ManagedEngine
    {
        private class Shard
        {
            public SliceCollection slices = new SliceCollection();
            public Thread thread;

            public Shard(string name)
            {
                thread = new Thread(ThreadWorker) { Name = name };
                thread.Start(null);
            }

            private static void ThreadWorker(object threadContext)
            {
            }
        }
    }
}
