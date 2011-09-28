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
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FeaServer.Engine.Tests.Mocks;

namespace FeaServer.Engine.Tests
{
    [TestClass]
    public class EngineFactoryUnitTests
    {
        [TestMethod]
        public void When_EngineProvider_Is_Cpu()
        {
            var elementTypes = new[] { new MockElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.Cpu))
            {
            }
        }

        [TestMethod]
        public void When_EngineProvider_Is_Cuda()
        {
            var elementTypes = new[] { new MockElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.Cuda))
            {
            }
        }

        [TestMethod]
        public void When_EngineProvider_Is_Managed()
        {
            var elementTypes = new[] { new MockElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.Managed))
            {
            }
        }

        [TestMethod]
        public void When_EngineProvider_Is_OpenCL_App()
        {
            var elementTypes = new[] { new MockElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.OpenCL_App))
            {
            }
        }

        [TestMethod]
        public void When_EngineProvider_Is_OpenCL_Cuda()
        {
            var elementTypes = new[] { new MockElementType() };
            using (var engine = EngineFactory.Create(elementTypes, EngineProvider.OpenCL_Cuda))
            {
            }
        }
    }
}
