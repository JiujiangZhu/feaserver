using System;
using System.Threading;
using System.Collections.Generic;
using System.Collections;
namespace FeaServer.Engine.Core
{
    /// <summary>
    /// FrugalThreadPool
    /// </summary>
    /// http://www.albahari.com/threading/part4.aspx
    public class FrugalThreadPool : Disposeable
    {
        private bool _disposed = false;
        private Thread[] _threadPool;
        private object[] _threadContext;
        private Queue<IEnumerable> _workQueue = new Queue<IEnumerable>();
        private ThreadStatus _threadStatus = ThreadStatus.Idle;
        private int _joiningThreadPoolCount;
        private object _joiningObject = new object();
        private Func<object> _threadContextBuilder;
        private Action<object, object> _executor;

        /// <summary>
        /// ThreadStatus
        /// </summary>
        private enum ThreadStatus
        {
            /// <summary>
            /// Idle
            /// </summary>
            Idle,
            /// <summary>
            /// Join
            /// </summary>
            Join,
            /// <summary>
            /// Stop
            /// </summary>
            Stop,
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="FrugalThreadPool2"/> class.
        /// </summary>
        public FrugalThreadPool(Action<object, object> executor)
            : this(4, executor, null) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="FrugalThreadPool2"/> class.
        /// </summary>
        /// <param name="threadCount">The thread count.</param>
        public FrugalThreadPool(int threadCount, Action<object, object> executor)
            : this(threadCount, executor, null) { }
        /// <summary>
        /// Initializes a new instance of the <see cref="FrugalThreadPool2"/> class.
        /// </summary>
        /// <param name="threadCount">The thread count.</param>
        /// <param name="threadContext">The thread context.</param>
        public FrugalThreadPool(int threadCount, Action<object, object> executor, Func<object> threadContextBuilder)
        {
            if (_executor == null)
                throw new ArgumentNullException("executor");
            _executor = executor;
            _threadPool = new Thread[threadCount];
            _threadContext = new object[threadCount];
            _threadContextBuilder = threadContextBuilder;
            for (int threadIndex = 0; threadIndex < _threadPool.Length; threadIndex++)
            {
                object threadContext;
                _threadPool[threadIndex] = CreateAndStartThread("FrugalPool: " + threadIndex.ToString(), out threadContext);
                _threadContext[threadIndex] = threadContext;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                Dispose(true);
            }
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                lock (this)
                {
                    _threadStatus = ThreadStatus.Stop;
                    Monitor.PulseAll(this);
                }
                foreach (Thread thread in _threadPool)
                    thread.Join();
            }
        }

        /// <summary>
        /// Creates the and start thread.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <returns></returns>
        private Thread CreateAndStartThread(string name, out object threadContext)
        {
            var thread = new Thread(ThreadWorker) { Name = name };
            threadContext = (_threadContextBuilder == null ? null : _threadContextBuilder());
            thread.Start(threadContext);
            return thread;
        }

        /// <summary>
        /// Gets the thread context.
        /// </summary>
        /// <value>The thread context.</value>
        public object[] ThreadContexts
        {
            get { return _threadContext; }
        }

        /// <summary>
        /// Threads the worker.
        /// </summary>
        private void ThreadWorker(object threadContext)
        {
            IEnumerable list;
            while (true)
            {
                lock (this)
                {
                    while (_workQueue.Count == 0)
                    {
                        switch (_threadStatus)
                        {
                            case ThreadStatus.Stop:
                                return;
                            case ThreadStatus.Join:
                                lock (_joiningObject)
                                {
                                    _joiningThreadPoolCount--;
                                    Monitor.Pulse(_joiningObject);
                                }
                                break;
                        }
                        Monitor.Wait(this);
                    }
                    list = _workQueue.Dequeue();
                }
                if (list != null)
                    foreach (object obj in list)
                        _executor(obj, threadContext);
            }
        }

        /// <summary>
        /// Adds the specified list.
        /// </summary>
        /// <param name="list">The list.</param>
        public void Add(IEnumerable list)
        {
            if (_threadStatus != ThreadStatus.Idle)
                throw new InvalidOperationException();
            lock (this)
            {
                _workQueue.Enqueue(list);
                Monitor.Pulse(this);
            }
        }

        /// <summary>
        /// Joins this instance.
        /// </summary>
        public void Join()
        {
            lock (this)
            {
                _threadStatus = ThreadStatus.Join;
                _joiningThreadPoolCount = _threadPool.Length;
                Monitor.PulseAll(this);
            }
            lock (_joiningObject)
            {
                while (_joiningThreadPoolCount > 0)
                    Monitor.Wait(_joiningObject);
                _threadStatus = ThreadStatus.Idle;
            }
        }

        /// <summary>
        /// Joins the and change.
        /// </summary>
        /// <param name="executor">The executor.</param>
        public void JoinAndChange(Action<object, object> executor)
        {
            Join();
            _executor = executor;
        }
    }
}
