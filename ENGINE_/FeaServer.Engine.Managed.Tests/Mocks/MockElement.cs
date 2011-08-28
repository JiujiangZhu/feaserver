using System;
using FeaServer.Engine;
namespace FeaServer.Tests.Mocks
{
    public class MockElement : IElement
    {
        public MockElement(IElementType type)
        {
            Type = type;
        }

        public IElementType Type { get; private set; }
    }
}
