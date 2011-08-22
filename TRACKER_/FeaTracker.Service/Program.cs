using System;
using System.ServiceModel;
using System.ServiceProcess;
using FeaTracker.Services;
namespace FeaTracker
{
    static class Program
    {
        private static ServiceHost _serviceHost;

        static void Main()
        {
            #region Run
            using (ExtensionService = new ExtensionService(TrackerService.Broadcast))
                if (!Environment.UserInteractive)
                    ServiceBase.Run(new[] { new FeaTrackerService() });
                else
                {
                    Initialize();
                    try
                    {
                        OnStart(null);
                        Console.WriteLine("All systems started.");
                        Console.WriteLine("Press <ENTER> to terminate service.");
                        Console.WriteLine();
                        Console.ReadKey();
                        Console.WriteLine();
                        Console.WriteLine("Stopping...");
                    }
                    finally { OnStop(); }
                }

            #endregion
        }

        public static IExtensionService ExtensionService { get; private set; }

        internal static void Initialize()
        {
            Console.ForegroundColor = ConsoleColor.White;
            ExtensionService.Initialize();
        }

        internal static void OnStart(string[] args)
        {
            ConsoleColor lastColor;

            // ServiceHost:
            Console.WriteLine("ServiceHost:");
            lastColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            _serviceHost = new ServiceHost(typeof(TrackerService));
            foreach (var endpoint in _serviceHost.Description.Endpoints)
            {
                Console.WriteLine("Endpoint - address:  {0}", endpoint.Address);
                Console.WriteLine("         - binding name:\t\t{0}", endpoint.Binding.Name);
                Console.WriteLine("         - contract name:\t\t{0}", endpoint.Contract.Name);
                Console.WriteLine();
            }
            Console.ForegroundColor = lastColor;
            _serviceHost.Open();
            //catch (TimeoutException timeProblem) { Console.WriteLine(timeProblem.Message); }
            //catch (CommunicationException commProblem) { Console.WriteLine(commProblem.Message); }
        }

        internal static void OnStop()
        {
            if ((_serviceHost != null) && (_serviceHost.State != CommunicationState.Closed))
                _serviceHost.Close();
        }
    }
}
