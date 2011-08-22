using System;
using System.Linq;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using FeaTracker.Models;
namespace FeaTracker.Services.Extension
{
    [Export]
    public class ExtensionFactory
    {
        [Import]
        private ITrackerInfoRepository _trackerInfoRepository = null;
    }
}
