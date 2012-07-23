using Contoso.IO;
using Pgno = System.UInt32;
namespace Contoso.Core
{
    public class Wal
    {
#if SQLITE_OMIT_WAL
        // was:sqlite3WalOpen
        internal static RC Open(VirtualFileSystem x, VirtualFile y, string z) { return RC.OK; }
#else
const int WAL_SAVEPOINT_NDATA = 4;
typedef struct Wal Wal;
int sqlite3WalOpen(VirtualFileSystem*, VirtualFile*, string , int, i64, Wal*);
int sqlite3WalClose(Wal *pWal, int sync_flags, int, u8 );
void sqlite3WalLimit(Wal*, i64);
int sqlite3WalBeginReadTransaction(Wal *pWal, int );
void sqlite3WalEndReadTransaction(Wal *pWal);
int sqlite3WalRead(Wal *pWal, Pgno pgno, int *pInWal, int nOut, u8 *pOut);
Pgno sqlite3WalDbsize(Wal *pWal);
int sqlite3WalBeginWriteTransaction(Wal *pWal);
int sqlite3WalEndWriteTransaction(Wal *pWal);
int sqlite3WalUndo(Wal *pWal, int (*xUndo)(void *, Pgno), object  *pUndoCtx);
void sqlite3WalSavepoint(Wal *pWal, u32 *aWalData);
int sqlite3WalSavepointUndo(Wal *pWal, u32 *aWalData);
int sqlite3WalFrames(Wal *pWal, int, PgHdr *, Pgno, int, int);
int sqlite3WalCheckpoint(
  Wal *pWal,                      /* Write-ahead log connection */
  int eMode,                      /* One of PASSIVE, FULL and RESTART */
  int (*xBusy)(void),            /* Function to call when busy */
  void *pBusyArg,                 /* Context argument for xBusyHandler */
  int sync_flags,                 /* Flags to sync db file with (or 0) */
  int nBuf,                       /* Size of buffer nBuf */
  u8 *zBuf,                       /* Temporary buffer to use */
  int *pnLog,                     /* OUT: Number of frames in WAL */
  int *pnCkpt                     /* OUT: Number of backfilled frames in WAL */
);
int sqlite3WalCallback(Wal *pWal);
int sqlite3WalExclusiveMode(Wal *pWal, int op);
int sqlite3WalHeapMemory(Wal *pWal);
#endif
    }

#if SQLITE_OMIT_WAL
    // was:sqlite3Wal*
    public static class WalExtensions
    {
        internal static void Limit(this Wal a, long y) { }
        internal static RC Close(this Wal a, int x, int y, byte z) { return 0; }
        internal static RC BeginReadTransaction(this Wal a, int z) { return 0; }
        internal static void EndReadTransaction(this Wal a) { }
        internal static RC Read(this Wal a, Pgno w, ref int x, int y, byte[] z) { return 0; }
        internal static Pgno DBSize(this Wal a) { return 0; }
        internal static RC BeginWriteTransaction(this Wal a) { return 0; }
        internal static RC EndWriteTransaction(this Wal a) { return 0; }
        internal static RC Undo(this Wal a, int y, object z) { return 0; }
        internal static void Savepoint(this Wal a, object z) { }
        internal static RC SavepointUndo(this Wal a, object z) { return 0; }
        internal static RC Frames(this Wal a, int v, PgHdr w, Pgno x, int y, int z) { return 0; }
        internal static RC Checkpoint(this Wal a, int s, int t, byte[] u, int v, int w, byte[] x, ref int y, ref int z) { y = 0; z = 0; return 0; }
        internal static RC Callback(this Wal a) { return 0; }
        internal static bool ExclusiveMode(this Wal a, int z) { return false; }
        internal static bool HeapMemory(this Wal a) { return false; }
    }
#endif
}