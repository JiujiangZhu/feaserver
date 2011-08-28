using System;
using FeaServer.Engine;
namespace Example.Mocks
{
    public class MockElementType : IElementType
    {
        public string Name
        {
            get { return "Sample"; }
        }

        public ElementScheduleStyle ScheduleStyle
        {
            get { return ElementScheduleStyle.FirstWins; }
        }

        public ElementImage GetImage(EngineProvider provider)
        {
            return new ElementImage { };
        }
    }
}
