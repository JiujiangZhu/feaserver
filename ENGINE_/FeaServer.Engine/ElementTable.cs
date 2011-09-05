using System;
using System.Collections.Generic;
namespace FeaServer.Engine
{
    public class ElementTable
    {
        public IEnumerable<IElement> Elements { get; set; }
        public IEnumerable<IElement> Links { get; set; }
    }
}
