using System;
using System.Reflection;
using Microsoft.Win32;
using System.Collections.Generic;
using System.IO;
namespace FeaServer.Engine
{
    public static class EngineFactory
    {
        public static IEngine Create() { return Create(null, GetTightestProvider()); }
        public static IEngine Create(IEnumerable<IElementType> elementTypes) { return Create(elementTypes, GetTightestProvider()); }
        public static IEngine Create(IEnumerable<IElementType> elementTypes, EngineProvider provider)
        {
            var enginePath = Environment.CurrentDirectory + "\\Engines\\";
            if (!Directory.Exists(enginePath))
                Directory.CreateDirectory(enginePath);
            Assembly assembly;
            Type engineType;
            switch (provider)
            {
                case EngineProvider.CUDA:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.Cuda.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.CudaEngine", true);
                    break;
                case EngineProvider.ATIStreams:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.AtiStreams.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.CalEngine", true);
                    break;
                case EngineProvider.CPU:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.Cpu.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.CpuEngine", true);
                    break;
                case EngineProvider.Managed:
                default:
                    throw new NotSupportedException();
            }
            if (engineType == null)
                throw new InvalidOperationException();
            var engine = (Activator.CreateInstance(engineType) as IEngine);
            if (elementTypes != null)
            {
                var engineElementTypes = engine.ElementTypes;
                foreach (var elementType in elementTypes)
                    engineElementTypes.Add(elementType);
            }
            return engine;
        }

        public static EngineProvider GetTightestProvider()
        {
            if (GetHasCUDA())
                return EngineProvider.CUDA;
            if (GetHasOpenCL())
                return EngineProvider.OpenCL;
            if (GetHasCal())
                return EngineProvider.ATIStreams;
            return EngineProvider.Managed;
        }

        private static bool GetHasCUDA()
        {
            string cudaPath = (Registry.GetValue(@"HKEY_LOCAL_MACHINE\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA", "RootInstallDir", null) as string);
            return (!string.IsNullOrEmpty(cudaPath));
        }

        private static bool GetHasCal()
        {
            return false;
        }

        private static bool GetHasOpenCL()
        {
            return false;
        }
    }
}
