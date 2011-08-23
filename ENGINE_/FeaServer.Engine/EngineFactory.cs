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
                case EngineProvider.Cpu:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.Cpu.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.CpuEngine", true);
                    break;
                case EngineProvider.Cuda:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.Cuda.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.CudaEngine", true);
                    break;
                case EngineProvider.Managed:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.Managed.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.ManagedEngine", true);
                    break;
                case EngineProvider.OpenCL_App:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.OpenCL.App.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.OpenCLEngine", true);
                    break;
                case EngineProvider.OpenCL_Cuda:
                    assembly = Assembly.LoadFile(Path.Combine(enginePath, "FeaServer.Engine.OpenCL.Cuda.dll"));
                    engineType = assembly.GetType("FeaServer.Engine.OpenCLEngine", true);
                    break;
                default:
                    throw new NotImplementedException();
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
                return EngineProvider.Cuda;
            if (GetHasOpenCL())
                return EngineProvider.OpenCL_Cuda;
            return EngineProvider.Cpu;
        }

        private static bool GetHasCUDA()
        {
            string cudaPath = (Registry.GetValue(@"HKEY_LOCAL_MACHINE\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA", "RootInstallDir", null) as string);
            return (!string.IsNullOrEmpty(cudaPath));
        }

        private static bool GetHasOpenCL()
        {
            return false;
        }
    }
}
