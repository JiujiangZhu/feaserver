using System;
namespace Contoso
{
    public static class SysEx
    {
        public const int SQLITE_VERSION_NUMBER = 300700701;

        public static void UNUSED_PARAMETER<T>(T x) { }
        public static void UNUSED_PARAMETER2<T1, T2>(T1 x, T2 y) { UNUSED_PARAMETER(x); UNUSED_PARAMETER(y); }

#if DEBUG || TRACE
        internal static bool OSTrace = false;
        internal static void OSTRACE(string x, params object[] args) { if (OSTrace) Console.WriteLine("a:" + string.Format(x, args)); }
#else
        internal static void OSTRACE(string x, params object[] args) { }
#endif

#if IOTRACE
        internal static bool IOTrace = true;
        public static void IOTRACE(string x, params object[] args) { if (IOTrace) Console.WriteLine("i:" + string.Format(x, args)); }
#else
        public static void IOTRACE(string x, params object[] args) { }
#endif

        public static int ROUND8(int x) { return (x + 7) & ~7; }
        public static int ROUNDDOWN8(int x) { return x & ~7; }

#if DEBUG
        public static RC SQLITE_CORRUPT_BKPT() { return sqlite3CorruptError(0); }
        public static RC SQLITE_MISUSE_BKPT() { return sqlite3MisuseError(0); }
        public static RC SQLITE_CANTOPEN_BKPT() { return sqlite3CantopenError(0); }

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
        //public static void sqlite3MemdebugSetType<T>(T X, MEMTYPE Y);
        //public static bool sqlite3MemdebugHasType<T>(T X, MEMTYPE Y);
        //public static bool sqlite3MemdebugNoType<T>(T X, MEMTYPE Y);
#else
        public static void sqlite3MemdebugSetType<T>(T X, MEMTYPE Y) { }
        public static bool sqlite3MemdebugHasType<T>(T X, MEMTYPE Y) { return true; }
        public static bool sqlite3MemdebugNoType<T>(T X, MEMTYPE Y) { return true; }
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
