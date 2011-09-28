#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
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
        public static IEngine Create(IEnumerable<IElementType> types, EngineProvider provider)
        {
            //var enginePath = Environment.CurrentDirectory + "\\Engines\\";
            var enginePath = @"C:\_APPLICATION\FEASERVER\ENGINE_\Engines\bin\x64\Debug";
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
            if (types != null)
            {
                var engineTypes = engine.Types;
                foreach (var type in types)
                    engineTypes.Add(type);
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
