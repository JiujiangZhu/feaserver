using System;
namespace FeaServer.Core
{
    [Flags]
    public enum FuncDefFlags : byte
    {
        LIKE = 0x01, // Candidate for the LIKE optimization
        CASE = 0x02, // Case-sensitive LIKE-type function
        EPHEM = 0x04, // Ephemeral.  Delete with VDBE
        NEEDCOLL = 0x08, // sqlite3GetFuncCollSeq() might be called
        COUNT = 0x20, // Built-in count(*) aggregate
        COALESCE = 0x40, // Built-in coalesce() or ifnull() function
    }
}
