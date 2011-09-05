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
        [TestMethod]
        public void AddZ_FirstWins_Moves_To_First_Element()
        {
            var time0 = new ElementCollection().xtor();
            var element0 = new Element(MockElement.FirstWins);
            time0.Add(element0, 0);
            //
            var time1 = new ElementCollection().xtor();
            var element1 = new Element(MockElement.FirstWins);
            time1.Add(element1, 0);

            Assert.AreEqual(time0.Count, 1);
            Assert.AreEqual(time1.Count, 0);
        }

        [TestMethod]
        public void AddZ_LastWins_Moves_To_Last_Element()
        {
            var time0 = new ElementCollection().xtor();
            var element0 = new Element(MockElement.LastWins);
            time0.Add(element0, 0);
            //
            var time1 = new ElementCollection().xtor();
            var element1 = new Element(MockElement.LastWins);
            time1.Add(element1, 0);

            Assert.AreEqual(time0.Count, 0);
            Assert.AreEqual(time1.Count, 1);
        }


        [TestMethod]
        public void AddZ_Multiple_Keeps_All_Elements()
        {
            var time0 = new ElementCollection().xtor();
            var element0 = new Element(MockElement.Multiple);
            time0.Add(element0, 0);
            //
            var time1 = new ElementCollection().xtor();
            var element1 = new Element(MockElement.Multiple);
            time1.Add(element1, 0);

            Assert.AreEqual(time0.Count, 1);
            Assert.AreEqual(time1.Count, 1);
        }
    }
}
