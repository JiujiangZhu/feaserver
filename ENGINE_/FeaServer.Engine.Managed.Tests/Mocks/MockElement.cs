using System;
using FeaServer.Engine;
namespace FeaServer.Tests.Mocks
{
    public class MockElement : IElement
    {
        public readonly static MockElement FirstWins = new MockElement(MockElementType.FirstWinsType);
        public readonly static MockElement LastWins = new MockElement(MockElementType.LastWinsType);
        public readonly static MockElement Multiple = new MockElement(MockElementType.MultipleType);

        public MockElement(IElementType type)
        {
            Type = type;
        }

        public IElementType Type { get; private set; }
    }
}
