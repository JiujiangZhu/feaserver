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
            var time0 = new ElementCollection(); time0.xtor();
            var element0 = new Element { ScheduleStyle = ElementScheduleStyle.FirstWins }; //(MockElement.FirstWins);
            time0.Add(element0, 0);
            //
            var time1 = new ElementCollection(); time1.xtor();
            var element1 = new Element { ScheduleStyle = ElementScheduleStyle.FirstWins }; //(MockElement.FirstWins);
            time1.Add(element1, 0);

            Assert.AreEqual(time0.Count, 1);
            Assert.AreEqual(time1.Count, 0);
        }

        [TestMethod]
        public void AddZ_LastWins_Moves_To_Last_Element()
        {
            var time0 = new ElementCollection(); time0.xtor();
            var element0 = new Element { ScheduleStyle = ElementScheduleStyle.LastWins }; //(MockElement.LastWins);
            time0.Add(element0, 0);
            //
            var time1 = new ElementCollection(); time1.xtor();
            var element1 = new Element { ScheduleStyle = ElementScheduleStyle.LastWins }; //(MockElement.LastWins);
            time1.Add(element1, 0);

            Assert.AreEqual(time0.Count, 0);
            Assert.AreEqual(time1.Count, 1);
        }


        [TestMethod]
        public void AddZ_Multiple_Keeps_All_Elements()
        {
            var time0 = new ElementCollection(); time0.xtor();
            var element0 = new Element { ScheduleStyle = ElementScheduleStyle.Multiple }; //(MockElement.Multiple);
            time0.Add(element0, 0);
            //
            var time1 = new ElementCollection(); time1.xtor();
            var element1 = new Element { ScheduleStyle = ElementScheduleStyle.Multiple }; //(MockElement.Multiple);
            time1.Add(element1, 0);

            Assert.AreEqual(time0.Count, 1);
            Assert.AreEqual(time1.Count, 1);
        }
    }
}
