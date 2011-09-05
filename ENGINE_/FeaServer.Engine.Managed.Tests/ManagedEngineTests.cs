using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FeaServer.Engine;
using FeaServer.Tests.Mocks;
namespace FeaServer.Tests
{
    [TestClass]
    public class ManagedEngineTests
    {
        [TestMethod]
        public void Integration()
        {
            using (var engine = new ManagedEngine())
            {
                // load types
                engine.Types.Add(MockElementType.FirstWinsType);
                engine.Types.Add(MockElementType.MultipleType);

                // load grids
                engine.LoadTable(new ElementTable
                {
                    Elements = new[] { MockElement.FirstWins }
                }, 0);
                engine.LoadTable(new ElementTable
                {
                    Elements = new[] { MockElement.FirstWins }
                }, 1);

                // evaluate frame
                engine.EvaluateFrame(TimePrec.EncodeTime(10));
            }
        }
    }
}
