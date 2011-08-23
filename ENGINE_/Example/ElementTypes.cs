using System;
using FeaServer.Engine;
namespace Example
{
    public class SampleElementType : IElementType
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
