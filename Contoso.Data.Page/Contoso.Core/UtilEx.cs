using System;
using Contoso.Threading;
namespace Contoso
{
    public static class UtilEx
    {
        private static readonly Random _r = new Random();

        public static void sqlite3_randomness(int N, ref long pBuf)
        {
            var zBuf = new byte[N];
            pBuf = 0;
#if SQLITE_THREADSAFE
            var mutex = MutexEx.sqlite3MutexAlloc(MutexEx.MUTEX.STATIC_PRNG);
#endif
            MutexEx.sqlite3_mutex_enter(mutex);
            while (N-- > 0)
                pBuf = (uint)((pBuf << 8) + (byte)_r.Next(byte.MaxValue));
            MutexEx.sqlite3_mutex_leave(mutex);
        }

        public static void sqlite3_randomness(byte[] pBuf, int Offset, int N)
        {
            var iBuf = DateTime.Now.Ticks;
#if SQLITE_THREADSAFE
            var mutex = MutexEx.sqlite3MutexAlloc(MutexEx.MUTEX.STATIC_PRNG);
#endif
            MutexEx.sqlite3_mutex_enter(mutex);
            while (N-- > 0)
            {
                iBuf = (uint)((iBuf << 8) + (byte)_r.Next(byte.MaxValue));
                pBuf[Offset++] = (byte)iBuf;
            }
            MutexEx.sqlite3_mutex_leave(mutex);
        }
    }
}
