using System;
using FeaServer.Engine;
namespace FeaServer.Tests.Mocks
{
    public class MockElementType : IElementType
    {
        public string Name { get; set; }

        public ElementScheduleStyle ScheduleStyle { get; set; }

        public ElementImage GetImage(EngineProvider provider)
        {
            return new ElementImage { };
        }
    }
}
