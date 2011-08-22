using System;
using System.Linq;
using System.ComponentModel.Composition.Hosting;
using System.Collections.Generic;
using System.IO;
using FeaTracker.Services.Extension;
using System.Reflection;
using System.ComponentModel.Composition;
using FeaTracker.Services.Tracker;
namespace FeaTracker.Services
{
    /// <summary>
    /// IExtensionService
    /// </summary>
    public interface IExtensionService : IDisposable
    {
        Action<Action<ITrackerServiceCallback>, Predicate<Subscriber>> Broadcast { get; }
        void Initialize();
    }

    /// <summary>
    /// ExtensionService
    /// </summary>
    public class ExtensionService : IExtensionService
    {
        private static ExtensionFactory _factory;

        public ExtensionService(Action<Action<ITrackerServiceCallback>, Predicate<Subscriber>> callback)
        {
            Broadcast = callback;
            Container = GetContainer();
        }
        public void Dispose()
        {
            Container.Dispose(); Container = null;
        }

        public Action<Action<ITrackerServiceCallback>, Predicate<Subscriber>> Broadcast { get; private set; }

        public void Initialize()
        {
            _factory = Container.GetExportedValue<ExtensionFactory>();
        }

        #region Container

        public static CompositionContainer Container { get; private set; }

        private static CompositionContainer GetContainer()
        {
            var pluginPath = Environment.CurrentDirectory + "\\Plugins\\";
            if (!Directory.Exists(pluginPath))
                Directory.CreateDirectory(pluginPath);
            var directoryCatalog = new DirectoryCatalog(pluginPath);
            var assemblyCatalog = new AssemblyCatalog(Assembly.GetExecutingAssembly());
            return new CompositionContainer(new AggregateCatalog(assemblyCatalog, directoryCatalog));
        }

        #endregion
    }
}
