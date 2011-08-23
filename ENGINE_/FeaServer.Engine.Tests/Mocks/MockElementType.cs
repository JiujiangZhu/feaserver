using System;
using FeaServer.Engine;
namespace FeaServer.Engine.Tests.Mocks
{
    public class MockElementType : IElementType
    {
        public string Name
        {
            get { return "Sample"; }
        }

        public ElementImage GetImage(EngineProvider provider)
        {
            return new ElementImage
            {
                StateBytes = 10,
                Image = null,
            };
        }
    }
}
