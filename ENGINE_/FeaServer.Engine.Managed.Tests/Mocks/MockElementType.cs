using System;
using FeaServer.Engine;
namespace FeaServer.Tests.Mocks
{
    public class MockElementType : IElementType
    {
        public readonly static MockElementType FirstWinsType = new MockElementType { Name = "F", ScheduleStyle = ElementScheduleStyle.FirstWins };
        public readonly static MockElementType LastWinsType = new MockElementType { Name = "L", ScheduleStyle = ElementScheduleStyle.LastWins };
        public readonly static MockElementType MultipleType = new MockElementType { Name = "M", ScheduleStyle = ElementScheduleStyle.Multiple };

        public string Name { get; set; }

        public ElementScheduleStyle ScheduleStyle { get; set; }

        public ElementImage GetImage(EngineProvider provider)
        {
            return new ElementImage { };
        }
    }
}
