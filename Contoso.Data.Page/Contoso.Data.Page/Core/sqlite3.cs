using Contoso.Sys;
using System;
namespace Contoso.Core
{
    public class sqlite3
    {
        [Flags]
        public enum SQLITE
        {
            VdbeTrace = 0x00000100,
            InternChanges = 0x00000200,
            FullColNames = 0x00000400,
            ShortColNames = 0x00000800,
            CountRows = 0x00001000,
            NullCallback = 0x00002000,
            SqlTrace = 0x00004000,
            VdbeListing = 0x00008000,
            WriteSchema = 0x00010000,
            NoReadlock = 0x00020000,
            IgnoreChecks = 0x00040000,
            ReadUncommitted = 0x0080000,
            LegacyFileFmt = 0x00100000,
            FullFSync = 0x00200000,
            CkptFullFSync = 0x00400000,
            RecoveryMode = 0x00800000,
            ReverseOrder = 0x01000000,
            RecTriggers = 0x02000000,
            ForeignKeys = 0x04000000,
            AutoIndex = 0x08000000,
            PreferBuiltin = 0x10000000,
            LoadExtension = 0x20000000,
            EnableTrigger = 0x40000000,
        }

        public sqlite3_mutex mutex { get; set; }
        public SQLITE flags { get; set; }
    }
}
