using System;
namespace Contoso.Core
{
    public interface ISchema
    {
        byte file_format { get; set; }
        SchemaFlags flags { get; set; }
        HashEx idxHash { get; set; }
    }

    [Flags]
    public enum SchemaFlags : byte
    {
        SchemaLoaded = 0x0001, // The schema has been loaded
        UnresetViews = 0x0002, // Some views have defined column names
        Empty = 0x0004, // The file is empty (length 0 bytes)
    }
}
