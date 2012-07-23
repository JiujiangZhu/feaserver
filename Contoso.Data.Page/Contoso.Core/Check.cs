using System.Diagnostics;
namespace Contoso
{
    public static class Check
    {
#if DEBUG
        public static bool ALWAYS(bool x) { if (x != true) Debug.Assert(false); return x; }
        public static int ALWAYS(int x) { if (x == 0) Debug.Assert(false); return x; }
        public static RC ALWAYS(RC x) { if (x == 0) Debug.Assert(false); return x; }
        public static T ALWAYS<T>(T x) { if (x == null) Debug.Assert(false); return x; }

        public static bool NEVER(bool x) { if (x == true) Debug.Assert(false); return x; }
        public static byte NEVER(byte x) { if (x != 0) Debug.Assert(false); return x; }
        public static int NEVER(int x) { if (x != 0) Debug.Assert(false); return x; }
        public static RC NEVER(RC x) { if (x != 0) Debug.Assert(false); return x; }
        public static T NEVER<T>(T x) { if (x != null) Debug.Assert(false); return x; }
#else
        public static bool ALWAYS(bool x) { return x; }
        public static byte ALWAYS(byte x) { return x; }
        public static int ALWAYS(int x) { return x; }
        public static RC ALWAYS(RC x) { return x; }
        public static bool ALWAYS<T>(T x) { return true; }

        public static bool NEVER(bool x) { return x; }
        public static byte NEVER(byte x) { return x; }
        public static int NEVER(int x) { return x; }
        public static RC NEVER(RC x) { return x; }
        public static bool NEVER<T>(T x) { return false; }
#endif

        public static bool LIKELY(bool x) { return !!x; }
        public static bool UNLIKELY(bool x) { return !!x; }
    }
}
