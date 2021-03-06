using System.Diagnostics;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
#if SQLITE_OMIT_SHARED_CACHE
        // The TableLock structure is only used by the sqlite3TableLock() and codeTableLocks() functions.
        public class TableLock
        {
            public int iDb;         /* The database containing the table to be locked */
            public int iTab;        /* The root page of the table to be locked */
            public byte isWriteLock;  /* True for write lock.  False for a read lock */
            public string zName;    /* Name of the table */
        }

        // Record the fact that we want to lock a table at run-time.
        //
        // The table to be locked has root page iTab and is found in database iDb. A read or a write lock can be taken depending on isWritelock.
        //
        // This routine just records the fact that the lock is desired.  The code to make the lock occur is generated by a later call to
        // codeTableLocks() which occurs during sqlite3FinishCoding().
        void sqlite3TableLock(Parse pParse, int iDb, int iTab, byte isWriteLock, string zName)
        {
            //Parse pToplevel = sqlite3ParseToplevel(pParse);
            //int nBytes;
            //TableLock p;
            //Debug.Assert(iDb >= 0);

            //for (int i = 0; i < pToplevel.nTableLock; i++)
            //{
            //    var p = pToplevel->aTableLock[i];
            //    if (p.iDb == iDb && p.iTab == iTab)
            //    {
            //        p->isWriteLock = (p.isWriteLock || isWriteLock);
            //        return;
            //    }
            //}

            //int nBytes = sizeof(vtableLock) * (pToplevel->nTableLock + 1);
            //pToplevel->aTableLock = sqlite3DbReallocOrFree(pToplevel->db, pToplevel->aTableLock, nBytes);
            //if (pToplevel->aTableLock)
            //{
            //    p = pToplevel->aTableLock[pToplevel->nTableLock++];
            //    p->iDb = iDb;
            //    p->iTab = iTab;
            //    p->isWriteLock = isWriteLock;
            //    p->zName = zName;
            //}
            //else
            //{
            //    pToplevel->nTableLock = 0;
            //    pToplevel->db->mallocFailed = 1;
            //}
        }

        // Code an OP_TableLock instruction for each table locked by the statement (configured by calls to sqlite3TableLock()).
        static void codeTableLocks(Parse pParse)
        {
            var pVdbe = sqlite3GetVdbe(pParse);
            Debug.Assert(pVdbe != null); // sqlite3GetVdbe cannot fail: VDBE already allocated
            for (int i = 0; i < pParse.nTableLock; i++)
            {
                TableLock p = pParse.aTableLock[i];
                int p1 = p.iDb;
                sqlite3VdbeAddOp4(pVdbe, OP_TableLock, p1, p.iTab, p.isWriteLock,
                p.zName, P4_STATIC);
            }
        }
#else
        static void codeTableLocks(Parse pParse) { }
#endif
    }
}