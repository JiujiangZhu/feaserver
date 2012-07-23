using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
namespace Contoso.IO
{
    public partial class VirtualFile
    {
        private static LockingStrategy lockingStrategy = (SysEx.IsRunningMediumTrust() ? new MediumTrustLockingStrategy() : new LockingStrategy());

        /// <summary>
        /// Basic locking strategy for Console/Winform applications
        /// </summary>
        private class LockingStrategy
        {
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
            [DllImport("kernel32.dll")]
            static extern bool LockFileEx(IntPtr hFile, uint dwFlags, uint dwReserved, uint nNumberOfBytesToLockLow, uint nNumberOfBytesToLockHigh, [In] ref System.Threading.NativeOverlapped lpOverlapped);
            const int LOCKFILE_FAIL_IMMEDIATELY = 1;
#endif
            public virtual void LockFile(VirtualFile pFile, long offset, long length)
            {
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                pFile.S.Lock(offset, length);
#endif
            }

            public virtual int SharedLockFile(VirtualFile pFile, long offset, long length)
            {
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                Debug.Assert(length == SHARED_SIZE);
                Debug.Assert(offset == SHARED_FIRST);
                var ovlp = new NativeOverlapped();
                ovlp.OffsetLow = (int)offset;
                ovlp.OffsetHigh = 0;
                ovlp.EventHandle = IntPtr.Zero;
                return (LockFileEx(pFile.S.Handle, LOCKFILE_FAIL_IMMEDIATELY, 0, (uint)length, 0, ref ovlp) ? 1 : 0);
#else
            return 1;
#endif
            }

            public virtual void UnlockFile(VirtualFile pFile, long offset, long length)
            {
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                pFile.S.Unlock(offset, length);
#endif
            }
        }

        /// <summary>
        /// Locking strategy for Medium Trust. It uses the same trick used in the native code for WIN_CE
        /// which doesn't support LockFileEx as well.
        /// </summary>
        private class MediumTrustLockingStrategy : LockingStrategy
        {
            public override int SharedLockFile(VirtualFile pFile, long offset, long length)
            {
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                Debug.Assert(length == SHARED_SIZE);
                Debug.Assert(offset == SHARED_FIRST);
                try  { pFile.S.Lock(offset + pFile.SharedLockByte, 1); }
                catch (IOException) {  return 0; }
#endif
                return 1;
            }
        }
    }
}