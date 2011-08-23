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
