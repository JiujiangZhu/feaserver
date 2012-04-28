using System.Diagnostics;
namespace FeaServer.Engine
{
    public static class Check
    {
#if !DEBUG
        internal static bool ALWAYS(bool X) { if (X != true) Debug.Assert(false); return true; }
        internal static int ALWAYS(int X) { if (X == 0) Debug.Assert(false); return 1; }
        internal static bool ALWAYS<T>(T X) { if (X == null) Debug.Assert(false); return true; }

        internal static bool NEVER(bool X) { if (X == true) Debug.Assert(false); return false; }
        internal static byte NEVER(byte X) { if (X != 0) Debug.Assert(false); return 0; }
        internal static int NEVER(int X) { if (X != 0) Debug.Assert(false); return 0; }
        internal static bool NEVER<T>(T X) { if (X != null) Debug.Assert(false); return false; }
#else
        internal static bool ALWAYS(bool X) { return X; }
        internal static byte ALWAYS(byte X) { return X; }
        internal static int ALWAYS(int X) { return X; }
        internal static bool ALWAYS<T>(T X) { return true; }

        internal static bool NEVER(bool X) { return X; }
        internal static byte NEVER(byte X) { return X; }
        internal static int NEVER(int X) { return X; }
        internal static bool NEVER<T>(T X) { return false; }
#endif
    }
}