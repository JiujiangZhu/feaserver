using System.Diagnostics;
using Contoso.Sys;
using System.Text;
using System;
namespace Contoso.Core
{
    public partial class FileEx
    {
        static FileEx()
        {
            sqlite3_vfs_register(new VirtualFileSystem(), 1);
        }

        internal static SQLITE sqlite3OsOpen(VirtualFileSystem pVfs, string zPath, VirtualFile pFile, VirtualFileSystem.OPEN flags, ref VirtualFileSystem.OPEN pFlagsOut)
        {
            // 0x87f3f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
            // SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
            var rc = pVfs.xOpen(zPath, pFile, (VirtualFileSystem.OPEN)((int)flags & 0x87f3f), out pFlagsOut); Debug.Assert(rc == SQLITE.OK || !pFile.isOpen); return rc;
        }
        //internal static SQLITE sqlite3OsDelete(VirtualFileSystem pVfs, string zPath, int dirSync) { return pVfs.xDelete(zPath, dirSync); }
        //internal static SQLITE sqlite3OsAccess(VirtualFileSystem pVfs, string zPath, VirtualFileSystem.ACCESS flags, ref int pResOut) { return pVfs.xAccess(zPath, flags, out pResOut); }
        //internal static SQLITE sqlite3OsFullPathname(VirtualFileSystem pVfs, string zPath, int nPathOut, StringBuilder zPathOut) { zPathOut.Length = 0; return pVfs.xFullPathname(zPath, nPathOut, zPathOut); }
#if !SQLITE_OMIT_LOAD_EXTENSION
        //internal static IntPtr sqlite3OsDlOpen(VirtualFileSystem pVfs, string zPath) { return pVfs.xDlOpen(zPath); }
        //internal static void sqlite3OsDlError(VirtualFileSystem pVfs, int nByte, string zBufOut) { pVfs.xDlError(nByte, zBufOut); }
        //internal static object sqlite3OsDlSym(VirtualFileSystem pVfs, IntPtr pHdle, ref string zSym) { return pVfs.xDlSym(pHdle, zSym); }
        //internal static void sqlite3OsDlClose(VirtualFileSystem pVfs, IntPtr pHandle) { pVfs.xDlClose(pHandle); }
#endif
        //internal static int sqlite3OsRandomness(VirtualFileSystem pVfs, int nByte, byte[] zBufOut) { return pVfs.xRandomness(nByte, zBufOut); }
        //internal static int sqlite3OsSleep(VirtualFileSystem pVfs, int nMicro) { return pVfs.xSleep(nMicro); }

        //internal static SQLITE sqlite3OsCurrentTimeInt64(VirtualFileSystem pVfs, ref long pTimeOut)
        //{
        //    // IMPLEMENTATION-OF: R-49045-42493 SQLite will use the xCurrentTimeInt64() method to get the current date and time if that method is available
        //    // (if iVersion is 2 or greater and the function pointer is not NULL) and will fall back to xCurrentTime() if xCurrentTimeInt64() is unavailable.
        //    SQLITE rc;
        //    if (pVfs.iVersion >= 2)
        //        rc = pVfs.xCurrentTimeInt64(ref pTimeOut);
        //    else
        //    {
        //        double r = 0;
        //        rc = pVfs.xCurrentTime(ref r);
        //        pTimeOut = (long)(r * 86400000.0);
        //    }
        //    return rc;
        //}

        internal static SQLITE sqlite3OsOpenMalloc(ref VirtualFileSystem pVfs, string zFile, ref VirtualFile ppFile, VirtualFileSystem.OPEN flags, ref VirtualFileSystem.OPEN pOutFlags)
        {
            var rc = SQLITE.NOMEM;
            var pFile = new VirtualFile();
            if (pFile != null)
            {
                rc = sqlite3OsOpen(pVfs, zFile, pFile, flags, ref pOutFlags);
                if (rc != SQLITE.OK)
                    pFile = null;
                else
                    ppFile = pFile;
            }
            return rc;
        }

        internal static int sqlite3OsInit() { return 0; }
        internal static VirtualFileSystem vfsList;
        internal static bool isInit = false;

        public static VirtualFileSystem sqlite3_vfs_find(string zVfs)
        {
            VirtualFileSystem pVfs = null;
            //#if !SQLITE_OMIT_AUTOINIT
            //            var rc = sqlite3_initialize();
            //            if (rc != SQLITE.OK)
            //                return null;
            //#endif
#if SQLITE_THREADSAFE
            var mutex = MutexEx.sqlite3MutexAlloc(MutexEx.MUTEX.STATIC_MASTER);
#endif
            MutexEx.sqlite3_mutex_enter(mutex);
            for (pVfs = vfsList; pVfs != null; pVfs = pVfs.pNext)
            {
                if (zVfs == null || zVfs == "") break;
                if (zVfs == pVfs.zName) break;
            }
            MutexEx.sqlite3_mutex_leave(mutex);
            return pVfs;
        }

        internal static void vfsUnlink(VirtualFileSystem pVfs)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(MutexEx.sqlite3MutexAlloc(MutexEx.MUTEX.STATIC_MASTER)));
            if (pVfs == null) { /* No-op */ }
            else if (vfsList == pVfs)
                vfsList = pVfs.pNext;
            else if (vfsList != null)
            {
                var p = vfsList;
                while (p.pNext != null && p.pNext != pVfs)
                    p = p.pNext;
                if (p.pNext == pVfs)
                    p.pNext = pVfs.pNext;
            }
        }

        public static SQLITE sqlite3_vfs_register(VirtualFileSystem pVfs, int makeDflt)
        {
            //#if !SQLITE_OMIT_AUTOINIT
            //            var rc = sqlite3_initialize();
            //            if (rc != SQLITE.OK)
            //                return rc;
            //#endif
            var mutex = MutexEx.sqlite3MutexAlloc(MutexEx.MUTEX.STATIC_MASTER);
            MutexEx.sqlite3_mutex_enter(mutex);
            vfsUnlink(pVfs);
            if (makeDflt != 0 || vfsList == null)
            {
                pVfs.pNext = vfsList;
                vfsList = pVfs;
            }
            else
            {
                pVfs.pNext = vfsList.pNext;
                vfsList.pNext = pVfs;
            }
            Debug.Assert(vfsList != null);
            MutexEx.sqlite3_mutex_leave(mutex);
            return SQLITE.OK;
        }

        internal static SQLITE sqlite3_vfs_unregister(VirtualFileSystem pVfs)
        {
#if SQLITE_THREADSAFE
            var mutex = MutexEx.sqlite3MutexAlloc(MutexEx.MUTEX.STATIC_MASTER);
#endif
            MutexEx.sqlite3_mutex_enter(mutex);
            vfsUnlink(pVfs);
            MutexEx.sqlite3_mutex_leave(mutex);
            return SQLITE.OK;
        }
    }
}
