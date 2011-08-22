using System;
using System.ServiceProcess;

namespace FeaTracker
{
    partial class FeaTrackerService : ServiceBase
    {
        public FeaTrackerService()
        {
            InitializeComponent();
            Program.Initialize();
        }

        protected override void OnStart(string[] args)
        {
            Program.OnStart(args);
        }

        protected override void OnStop()
        {
            Program.OnStop();
        }
    }
}
