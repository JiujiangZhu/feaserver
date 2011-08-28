using System;
using System.Collections.ObjectModel;
namespace FeaServer.Engine
{
    public interface IElementType
    {
        string Name { get; }
        ElementScheduleStyle ScheduleStyle { get; }
        ElementImage GetImage(EngineProvider provider);
    }
}
