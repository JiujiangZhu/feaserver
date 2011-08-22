using System;
using System.Collections.ObjectModel;
using System.Collections.Generic;
namespace FeaServer.Engine
{
    public class ElementTypeCollection : Collection<IElementType>
    {
        public ElementTypeCollection(IList<IElementType> list)
            : base(list) { }
    }
}
