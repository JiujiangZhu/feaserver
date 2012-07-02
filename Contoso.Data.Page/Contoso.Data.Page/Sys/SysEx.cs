using System;
namespace Contoso.Sys
{
    public static class SysEx
    {
        public const int SQLITE_VERSION_NUMBER = 300700701;

        internal static void UNUSED_PARAMETER<T>(T x) { }
        internal static void UNUSED_PARAMETER2<T1, T2>(T1 x, T2 y) { UNUSED_PARAMETER(x); UNUSED_PARAMETER(y); }

#if DEBUG || TRACE
        internal static bool sqlite3OsTrace = true;
        internal static void OSTRACE(string x, params object[] args) { if (sqlite3OsTrace) Console.WriteLine(string.Format(x, args)); }
#else
        internal static void OSTRACE(string x, params object[] args) { }
#endif

#if IOTRACE
        internal static bool SQLite3IoTrace = true;
        internal static void IOTRACE(string x, params object[] args) { if (SQLite3IoTrace) Console.WriteLine(string.Format(x, args)); }
#else
        internal static void IOTRACE(string x, params object[] args) { }
#endif

        internal static int ROUND8(int x) { return (x + 7) & ~7; }
        internal static int ROUNDDOWN8(int x) { return x & ~7; }

#if DEBUG
        internal static RC SQLITE_CORRUPT_BKPT() { return sqlite3CorruptError(0); }
        internal static RC SQLITE_MISUSE_BKPT() { return sqlite3MisuseError(0); }
        internal static RC SQLITE_CANTOPEN_BKPT() { return sqlite3CantopenError(0); }

        static RC sqlite3CorruptError(int lineno)
        {
            //sqlite3_log(SQLITE_CORRUPT, "database corruption at line %d of [%.10s]", lineno, 20 + sqlite3_sourceid());
            return RC.CORRUPT;
        }
        static RC sqlite3MisuseError(int lineno)
        {
            //sqlite3_log(SQLITE_MISUSE, "misuse at line %d of [%.10s]", lineno, 20 + sqlite3_sourceid());
            return RC.MISUSE;
        }
        static RC sqlite3CantopenError(int lineno)
        {
            //sqlite3_log(SQLITE_CANTOPEN, "cannot open file at line %d of [%.10s]", lineno, 20 + sqlite3_sourceid());
            return RC.CANTOPEN;
        }

#else
        internal static SQLITE SQLITE_CORRUPT_BKPT() { return SQLITE_CORRUPT; }
        internal static SQLITE SQLITE_MISUSE_BKPT() { return SQLITE_MISUSE; }
        internal static SQLITE SQLITE_CANTOPEN_BKPT() { return SQLITE_CANTOPEN; }
#endif

#if SQLITE_MEMDEBUG
        //internal static void sqlite3MemdebugSetType<T>(T X, MEMTYPE Y);
        //internal static bool sqlite3MemdebugHasType<T>(T X, MEMTYPE Y);
        //internal static bool sqlite3MemdebugNoType<T>(T X, MEMTYPE Y);
#else
        internal static void sqlite3MemdebugSetType<T>(T X, MEMTYPE Y) { }
        internal static bool sqlite3MemdebugHasType<T>(T X, MEMTYPE Y) { return true; }
        internal static bool sqlite3MemdebugNoType<T>(T X, MEMTYPE Y) { return true; }
#endif
        [Flags]
        public enum MEMTYPE
        {
            HEAP = 0x01,         // General heap allocations
            LOOKASIDE = 0x02,    // Might have been lookaside memory
            SCRATCH = 0x04,      // Scratch allocations
            PCACHE = 0x08,       // Page cache allocations
            DB = 0x10,           // Uses sqlite3DbMalloc, not sqlite_malloc
        }

        public static bool IsRunningMediumTrust() { return false; }

        public static T getGLOBAL<T>(T v) { return v; }
        public static void setGLOBAL<T>(T v, T value) { v = value; }
    }
}
