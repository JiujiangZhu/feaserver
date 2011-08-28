using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FeaServer.Engine.Time.Scheduler;
using FeaServer.Tests.Mocks;
using FeaServer.Engine;
namespace FeaServer.Tests.Scheduler
{
    [TestClass]
    public class ElementCollectionTests
    {
        private readonly static MockElementType MockFirstWinsType = new Mocks.MockElementType { Name = "F", ScheduleStyle = ElementScheduleStyle.FirstWins };
        private readonly static MockElementType MockLastWinsType = new Mocks.MockElementType { Name = "L", ScheduleStyle = ElementScheduleStyle.LastWins };
        private readonly static MockElementType MockMultipleType = new Mocks.MockElementType { Name = "M", ScheduleStyle = ElementScheduleStyle.Multiple };
        private readonly static MockElement MockFirstWins = new Mocks.MockElement(MockFirstWinsType);
        private readonly static MockElement MockLastWins = new Mocks.MockElement(MockLastWinsType);
        private readonly static MockElement MockMultiple = new Mocks.MockElement(MockMultipleType);

        [TestMethod]
        public void AddZ_FirstWins_Moves_To_First_Element()
        {
            var time0 = new ElementCollection();
            var element0 = new Element(MockFirstWins);
            time0.Add(element0, null);
            //
            var time1 = new ElementCollection();
            var element1 = new Element(MockFirstWins);
            time1.Add(element1, null);
        }

        [TestMethod]
        public void AddZ_LastWins_Moves_To_Last_Element()
        {
            var time0 = new ElementCollection();
            var element0 = new Element(MockLastWins);
            time0.Add(element0, null);
            //
            var time1 = new ElementCollection();
            var element1 = new Element(MockLastWins);
            time1.Add(element1, null);
        }


        [TestMethod]
        public void AddZ_Multiple_Keeps_All_Elements()
        {
            var time0 = new ElementCollection();
            var element0 = new Element(MockMultiple);
            time0.Add(element0, null);
            //
            var time1 = new ElementCollection();
            var element1 = new Element(MockMultiple);
            time1.Add(element1, null);
        }

    }
}
