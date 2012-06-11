using System;
using System.Diagnostics;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;
using SAVEPOINT = Contoso.Core.Pager.SAVEPOINT;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;

namespace Contoso.Core
{
    public partial class Btree
    {
#if TRACE
        internal static bool sqlite3BtreeTrace = true;  // True to enable tracing
        internal static void TRACE(string x, params object[] args) { if (sqlite3BtreeTrace)Console.WriteLine(string.Format(x, args)); }
#else
        internal static void TRACE(string x, params object[] args) { }
#endif

        #region Properties

        internal Pager sqlite3BtreePager()
        {
            return this.Shared.Pager;
        }

        internal string sqlite3BtreeGetFilename()
        {
            Debug.Assert(this.Shared.Pager != null);
            return this.Shared.Pager.sqlite3PagerFilename();
        }

        internal string sqlite3BtreeGetJournalname()
        {
            Debug.Assert(this.Shared.Pager != null);
            return this.Shared.Pager.sqlite3PagerJournalname();
        }

        internal bool sqlite3BtreeIsInTrans()
        {
            Debug.Assert(this == null || MutexEx.Held(this.DB.Mutex));
            return (this != null && (this.inTrans == TRANS.WRITE));
        }

        internal Pgno sqlite3BtreeLastPage()
        {
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(((this.Shared.Pages) & 0x8000000) == 0);
            return (Pgno)this.Shared.btreePagecount();
        }

        #endregion

        internal static void pageReinit(DbPage pData)
        {
            var pPage = Pager.sqlite3PagerGetExtra(pData);
            Debug.Assert(Pager.sqlite3PagerPageRefcount(pData) > 0);
            if (pPage.HasInit)
            {
                Debug.Assert(MutexEx.Held(pPage.Shared.Mutex));
                pPage.HasInit = false;
                if (Pager.sqlite3PagerPageRefcount(pData) > 1)
                    // pPage might not be a btree page;  it might be an overflow page or ptrmap page or a free page.  In those cases, the following
                    // call to btreeInitPage() will likely return SQLITE_CORRUPT. But no harm is done by this.  And it is very important that
                    // btreeInitPage() be called on every btree page so we make the call for every page that comes in for re-initing.
                    pPage.btreeInitPage();
            }
        }

        internal static int btreeInvokeBusyHandler(object pArg)
        {
            var pBt = (BtShared)pArg;
            Debug.Assert(pBt.DB != null);
            Debug.Assert(MutexEx.Held(pBt.DB.Mutex));
            return pBt.DB.sqlite3InvokeBusyHandler();
        }

        internal static RC sqlite3BtreeOpen(VirtualFileSystem pVfs, string zFilename, sqlite3 db, ref Btree ppBtree, OPEN flags, VFSOPEN vfsFlags)
        {
            BtShared pBt = null;          // Shared part of btree structure
            Btree p;                      // Handle to return
            sqlite3_mutex mutexOpen = null;  // Prevents a race condition.
            var rc = RC.OK;
            byte nReserve;                   // Byte of unused space on each page
            var zDbHeader = new byte[100]; // Database header content
            // True if opening an ephemeral, temporary database */
            bool isTempDb = string.IsNullOrEmpty(zFilename);
            // Set the variable isMemdb to true for an in-memory database, or  false for a file-based database.
#if SQLITE_OMIT_MEMORYDB
            var isMemdb = false;
#else
            var isMemdb = (zFilename == ":memory:" || isTempDb && db.sqlite3TempInMemory());
#endif
            Debug.Assert(db != null);
            Debug.Assert(pVfs != null);
            Debug.Assert(MutexEx.Held(db.Mutex));
            Debug.Assert(((uint)flags & 0xff) == (uint)flags);   // flags fit in 8 bits
            // Only a BTREE_SINGLE database can be BTREE_UNORDERED
            Debug.Assert((flags & OPEN.UNORDERED) == 0 || (flags & OPEN.SINGLE) != 0);
            // A BTREE_SINGLE database is always a temporary and/or ephemeral
            Debug.Assert((flags & OPEN.SINGLE) == 0 || isTempDb);
            if ((db.flags & sqlite3.SQLITE.NoReadlock) != 0)
                flags |= OPEN.NO_READLOCK;
            if (isMemdb)
                flags |= OPEN.MEMORY;
            if ((vfsFlags & VFSOPEN.MAIN_DB) != 0 && (isMemdb || isTempDb))
                vfsFlags = (vfsFlags & ~VFSOPEN.MAIN_DB) | VFSOPEN.TEMP_DB;
            p = new Btree();
            p.inTrans = TRANS.NONE;
            p.DB = db;
#if !SQLITE_OMIT_SHARED_CACHE
            p.Locks.pBtree = p;
            p.Locks.iTable = 1;
#endif
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
            // If this Btree is a candidate for shared cache, try to find an existing BtShared object that we can share with
            if (!isMemdb && !isTempDb)
            {
                if (vfsFlags & SQLITE_OPEN_SHAREDCACHE)
                {
                    int nFullPathname = pVfs.mxPathname + 1;
                    string zFullPathname = MallocEx.sqlite3Malloc(nFullPathname);
                    sqlite3_mutex mutexShared;
                    p.sharable = true;
                    if (zFullPathname == null)
                    {
                        p = null;
                        return RC.NOMEM;
                    }
                    sqlite3OsFullPathname(pVfs, zFilename, nFullPathname, zFullPathname);
                    mutexOpen = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_OPEN);
                    MutexEx.sqlite3_mutex_enter(mutexOpen);
                    mutexShared = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
                    MutexEx.sqlite3_mutex_enter(mutexShared);
                    for (pBt = GLOBAL(BtShared, sqlite3SharedCacheList); pBt != null; pBt = pBt.Next)
                    {
                        Debug.Assert(pBt.nRef > 0);
                        if (0 == strcmp(zFullPathname, sqlite3PagerFilename(pBt.Pager))
                        && sqlite3PagerVfs(pBt.Pager) == pVfs)
                        {
                            int iDb;
                            for (iDb = db.nDb - 1; iDb >= 0; iDb--)
                            {
                                Btree pExisting = db.aDb[iDb].pBt;
                                if (pExisting && pExisting.Shared == pBt)
                                {
                                    MutexEx.sqlite3_mutex_leave(mutexShared);
                                    MutexEx.sqlite3_mutex_leave(mutexOpen);
                                    zFullPathname = null;//sqlite3_free(ref zFullPathname);
                                    p = null;//sqlite3_free(ref p);
                                    return RC.CONSTRAINT;
                                }
                            }
                            p.Shared = pBt;
                            pBt.nRef++;
                            break;
                        }
                    }
                    MutexEx.sqlite3_mutex_leave(mutexShared);
                    zFullPathname = null;//sqlite3_free(ref zFullPathname);
                }
#if DEBUG
                else
                {
                    // In debug mode, we mark all persistent databases as sharable even when they are not.  This exercises the locking code and
                    // gives more opportunity for asserts(sqlite3_mutex_held()) statements to find locking problems.
                    p.sharable = true;
                }
#endif
            }
#endif
            if (pBt == null)
            {
                // The following asserts make sure that structures used by the btree are the right size.  This is to guard against size changes that result
                // when compiling on a different architecture.
                Debug.Assert(sizeof(long) == 8 || sizeof(long) == 4);
                Debug.Assert(sizeof(ulong) == 8 || sizeof(ulong) == 4);
                Debug.Assert(sizeof(uint) == 4);
                Debug.Assert(sizeof(ushort) == 2);
                Debug.Assert(sizeof(Pgno) == 4);
                pBt = new BtShared();
                rc = Pager.sqlite3PagerOpen(pVfs, out pBt.Pager, zFilename, EXTRA_SIZE, (Pager.PAGEROPEN)flags, vfsFlags, pageReinit);
                if (rc == RC.OK)
                    rc = pBt.Pager.sqlite3PagerReadFileheader(zDbHeader.Length, zDbHeader);
                if (rc != RC.OK)
                    goto btree_open_out;
                pBt.OpenFlags = flags;
                pBt.DB = db;
                pBt.Pager.sqlite3PagerSetBusyhandler(btreeInvokeBusyHandler, pBt);
                p.Shared = pBt;
                pBt.Cursors = null;
                pBt.Page1 = null;
                pBt.ReadOnly = pBt.Pager.sqlite3PagerIsreadonly();
#if SQLITE_SECURE_DELETE
pBt.secureDelete = true;
#endif
                pBt.PageSize = (uint)((zDbHeader[16] << 8) | (zDbHeader[17] << 16)); if (pBt.PageSize < 512 || pBt.PageSize > Pager.SQLITE_MAX_PAGE_SIZE || ((pBt.PageSize - 1) & pBt.PageSize) != 0)
                {
                    pBt.PageSize = 0;
#if !SQLITE_OMIT_AUTOVACUUM
                    // If the magic name ":memory:" will create an in-memory database, then leave the autoVacuum mode at 0 (do not auto-vacuum), even if
                    // SQLITE_DEFAULT_AUTOVACUUM is true. On the other hand, if SQLITE_OMIT_MEMORYDB has been defined, then ":memory:" is just a
                    // regular file-name. In this case the auto-vacuum applies as per normal.
                    if (zFilename != string.Empty && !isMemdb)
                    {
                        pBt.AutoVacuum = (AUTOVACUUM.DEFAULT != AUTOVACUUM.NONE);
                        pBt.IncrVacuum = (AUTOVACUUM.DEFAULT == AUTOVACUUM.INCR);
                    }
#endif
                    nReserve = 0;
                }
                else
                {
                    nReserve = zDbHeader[20];
                    pBt.PageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
                    pBt.AutoVacuum = ConvertEx.Get4(zDbHeader, 36 + 4 * 4) != 0;
                    pBt.IncrVacuum = ConvertEx.Get4(zDbHeader, 36 + 7 * 4) != 0;
#endif
                }
                rc = pBt.Pager.sqlite3PagerSetPagesize(ref pBt.PageSize, nReserve);
                if (rc != RC.OK)
                    goto btree_open_out;
                pBt.UsableSize = (ushort)(pBt.PageSize - nReserve);
                Debug.Assert((pBt.PageSize & 7) == 0);  // 8-byte alignment of pageSize
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
                // Add the new BtShared object to the linked list sharable BtShareds.
                if (p.sharable)
                {
                    sqlite3_mutex mutexShared;
                    pBt.nRef = 1;
                    mutexShared = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
                    if (SQLITE_THREADSAFE && sqlite3GlobalConfig.bCoreMutex)
                    {
                        pBt.mutex = sqlite3MutexAlloc(SQLITE_MUTEX_FAST);
                        if (pBt.mutex == null)
                        {
                            rc = SQLITE_NOMEM;
                            db.mallocFailed = 0;
                            goto btree_open_out;
                        }
                    }
                    sqlite3_mutex_enter(mutexShared);
                    pBt.pNext = GLOBAL(BtShared, sqlite3SharedCacheList);
                    GLOBAL(BtShared, sqlite3SharedCacheList) = pBt;
                    sqlite3_mutex_leave(mutexShared);
                }
#endif
            }
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
            /* If the new Btree uses a sharable pBtShared, then link the new
** Btree into the list of all sharable Btrees for the same connection.
** The list is kept in ascending order by pBt address.
*/
            if (p.sharable)
            {
                int i;
                Btree pSib;
                for (i = 0; i < db.nDb; i++)
                {
                    if ((pSib = db.aDb[i].pBt) != null && pSib.sharable)
                    {
                        while (pSib.pPrev) { pSib = pSib.pPrev; }
                        if (p.pBt < pSib.pBt)
                        {
                            p.pNext = pSib;
                            p.pPrev = 0;
                            pSib.pPrev = p;
                        }
                        else
                        {
                            while (pSib.pNext && pSib.pNext.pBt < p.pBt)
                            {
                                pSib = pSib.pNext;
                            }
                            p.pNext = pSib.pNext;
                            p.pPrev = pSib;
                            if (p.pNext)
                            {
                                p.pNext.pPrev = p;
                            }
                            pSib.pNext = p;
                        }
                        break;
                    }
                }
            }
#endif
            ppBtree = p;
        btree_open_out:
            if (rc != RC.OK)
            {
                if (pBt != null && pBt.Pager != null)
                    pBt.Pager.sqlite3PagerClose();
                pBt = null;
                p = null;
                ppBtree = null;
            }
            else
            {
                // If the B-Tree was successfully opened, set the pager-cache size to the default value. Except, when opening on an existing shared pager-cache,
                // do not change the pager-cache size.
                if (p.sqlite3BtreeSchema(0, null, null) == null)
                    p.Shared.Pager.sqlite3PagerSetCachesize(SQLITE_DEFAULT_CACHE_SIZE);
            }
            if (mutexOpen != null)
            {
                Debug.Assert(MutexEx.Held(mutexOpen));
                MutexEx.sqlite3_mutex_leave(mutexOpen);
            }
            return rc;
        }

        internal static RC sqlite3BtreeClose(ref Btree p)
        {
            // Close all cursors opened via this handle.
            Debug.Assert(MutexEx.Held(p.DB.Mutex));
            p.sqlite3BtreeEnter();
            var pBt = p.Shared;
            var pCur = pBt.Cursors;
            while (pCur != null)
            {
                var pTmp = pCur;
                pCur = pCur.Next;
                if (pTmp.Tree == p)
                    pTmp.Close();
            }
            // Rollback any active transaction and free the handle structure. The call to sqlite3BtreeRollback() drops any table-locks held by this handle.
            p.sqlite3BtreeRollback();
            p.sqlite3BtreeLeave();
            // If there are still other outstanding references to the shared-btree structure, return now. The remainder of this procedure cleans up the shared-btree.
            Debug.Assert(p.wantToLock == 0 && !p.locked);
            if (!p.sharable || pBt.removeFromSharingList())
            {
                // The pBt is no longer on the sharing list, so we can access it without having to hold the mutex.
                // Clean out and delete the BtShared object.
                Debug.Assert(null == pBt.Cursors);
                pBt.Pager.sqlite3PagerClose();
                if (pBt.xFreeSchema != null && pBt.Schema != null)
                    pBt.xFreeSchema(pBt.Schema);
                pBt.Schema = null;// sqlite3DbFree(0, pBt->pSchema);
                //freeTempSpace(pBt);
                pBt = null; //sqlite3_free(ref pBt);
            }
#if !SQLITE_OMIT_SHARED_CACHE
            Debug.Assert(p.wantToLock == null);
            Debug.Assert(p.locked == null);
            if (p.pPrev != null) p.pPrev.pNext = p.pNext;
            if (p.pNext != null) p.pNext.pPrev = p.pPrev;
#endif
            return RC.OK;
        }

        internal RC sqlite3BtreeSetCacheSize(int mxPage)
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            pBt.Pager.sqlite3PagerSetCachesize(mxPage);
            sqlite3BtreeLeave();
            return RC.OK;
        }

#if !SQLITE_OMIT_PAGER_PRAGMAS
        internal RC sqlite3BtreeSetSafetyLevel(int level, int fullSync, int ckptFullSync)
        {
            var pBt = this.Shared;
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            Debug.Assert(level >= 1 && level <= 3);
            sqlite3BtreeEnter();
            pBt.Pager.sqlite3PagerSetSafetyLevel(level, fullSync, ckptFullSync);
            sqlite3BtreeLeave();
            return RC.OK;
        }
#endif

        internal int sqlite3BtreeSyncDisabled()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            Debug.Assert(pBt != null && pBt.Pager != null);
            var rc = (pBt.Pager.sqlite3PagerNosync() ? 1 : 0);
            sqlite3BtreeLeave();
            return rc;
        }

        internal RC sqlite3BtreeSetPageSize(int pageSize, int nReserve, int iFix)
        {
            Debug.Assert(nReserve >= -1 && nReserve <= 255);
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            if (pBt.PageSizeFixed)
            {
                sqlite3BtreeLeave();
                return RC.READONLY;
            }
            if (nReserve < 0)
                nReserve = (int)(pBt.PageSize - pBt.UsableSize);
            Debug.Assert(nReserve >= 0 && nReserve <= 255);
            if (pageSize >= 512 && pageSize <= Pager.SQLITE_MAX_PAGE_SIZE && ((pageSize - 1) & pageSize) == 0)
            {
                Debug.Assert((pageSize & 7) == 0);
                Debug.Assert(pBt.Page1 == null && pBt.Cursors == null);
                pBt.PageSize = (uint)pageSize;
            }
            var rc = pBt.Pager.sqlite3PagerSetPagesize(ref pBt.PageSize, nReserve);
            pBt.UsableSize = (ushort)(pBt.PageSize - nReserve);
            if (iFix != 0)
                pBt.PageSizeFixed = true;
            sqlite3BtreeLeave();
            return rc;
        }

        internal int sqlite3BtreeGetPageSize() { return (int)this.Shared.PageSize; }

#if !SQLITE_OMIT_PAGER_PRAGMAS || !SQLITE_OMIT_VACUUM
        internal int sqlite3BtreeGetReserve()
        {
            sqlite3BtreeEnter();
            var n = (int)(this.Shared.PageSize - this.Shared.UsableSize);
            sqlite3BtreeLeave();
            return n;
        }

        internal Pgno sqlite3BtreeMaxPageCount(int mxPage)
        {
            sqlite3BtreeEnter();
            var n = (Pgno)this.Shared.Pager.sqlite3PagerMaxPageCount(mxPage);
            sqlite3BtreeLeave();
            return n;
        }

        internal int sqlite3BtreeSecureDelete(int newFlag)
        {
            if (this == null)
                return 0;
            sqlite3BtreeEnter();
            if (newFlag >= 0)
                this.Shared.SecureDelete = (newFlag != 0);
            var b = (this.Shared.SecureDelete ? 1 : 0);
            sqlite3BtreeLeave();
            return b;
        }
#endif

        internal RC sqlite3BtreeSetAutoVacuum(int autoVacuum)
        {
#if SQLITE_OMIT_AUTOVACUUM
            return SQLITE.READONLY;
#else
            sqlite3BtreeEnter();
            var rc = RC.OK;
            var av = (byte)autoVacuum;
            var pBt = this.Shared;
            if (pBt.PageSizeFixed && (av != 0) != pBt.AutoVacuum)
                rc = RC.READONLY;
            else
            {
                pBt.AutoVacuum = av != 0;
                pBt.IncrVacuum = av == 2;
            }
            sqlite3BtreeLeave();
            return rc;
#endif
        }

        internal AUTOVACUUM sqlite3BtreeGetAutoVacuum()
        {
#if SQLITE_OMIT_AUTOVACUUM
return BTREE.AUTOVACUUM_NONE;
#else
            sqlite3BtreeEnter();
            var rc = (!this.Shared.AutoVacuum ? AUTOVACUUM.NONE : (!this.Shared.IncrVacuum ? AUTOVACUUM.FULL : AUTOVACUUM.INCR));
            sqlite3BtreeLeave();
            return rc;
#endif
        }

        internal RC sqlite3BtreeBeginTrans(int wrflag)
        {
            var pBt = this.Shared;
            var rc = RC.OK;
            sqlite3BtreeEnter();
            btreeIntegrity();
            // If the btree is already in a write-transaction, or it is already in a read-transaction and a read-transaction is requested, this is a no-op.
            if (this.inTrans == TRANS.WRITE || (this.inTrans == TRANS.READ && 0 == wrflag))
                goto trans_begun;
            // Write transactions are not possible on a read-only database
            if (pBt.ReadOnly && wrflag != 0)
            {
                rc = RC.READONLY;
                goto trans_begun;
            }
#if !SQLITE_OMIT_SHARED_CACHE
            /* If another database handle has already opened a write transaction
** on this shared-btree structure and a second write transaction is
** requested, return SQLITE_LOCKED.
*/
            if ((wrflag && pBt.inTransaction == TRANS_WRITE) || pBt.isPending)
            {
                sqlite3 pBlock = pBt.pWriter.db;
            }
            else if (wrflag > 1)
            {
                BtLock pIter;
                for (pIter = pBt.pLock; pIter; pIter = pIter.pNext)
                {
                    if (pIter.pBtree != p)
                    {
                        pBlock = pIter.pBtree.db;
                        break;
                    }
                }
            }
            if (pBlock)
            {
                sqlite3ConnectionBlocked(p.db, pBlock);
                rc = SQLITE_LOCKED_SHAREDCACHE;
                goto trans_begun;
            }
#endif
            // Any read-only or read-write transaction implies a read-lock on page 1. So if some other shared-cache client already has a write-lock
            // on page 1, the transaction cannot be opened. */
            rc = querySharedCacheTableLock(this, MASTER_ROOT, BtreeLock.LOCK.READ);
            if (rc != RC.OK)
                goto trans_begun;
            pBt.InitiallyEmpty = pBt.Pages == 0;
            do
            {
                // Call lockBtree() until either pBt.pPage1 is populated or lockBtree() returns something other than SQLITE_OK. lockBtree()
                // may return SQLITE_OK but leave pBt.pPage1 set to 0 if after reading page 1 it discovers that the page-size of the database
                // file is not pBt.pageSize. In this case lockBtree() will update pBt.pageSize to the page-size of the file on disk.
                while (pBt.Page1 == null && (rc = pBt.lockBtree()) == RC.OK) ;
                if (rc == RC.OK && wrflag != 0)
                {
                    if (pBt.ReadOnly)
                        rc = RC.READONLY;
                    else
                    {
                        rc = pBt.Pager.sqlite3PagerBegin(wrflag > 1, this.DB.sqlite3TempInMemory() ? 1 : 0);
                        if (rc == RC.OK)
                            rc = pBt.newDatabase();
                    }
                }
                if (rc != RC.OK)
                    pBt.unlockBtreeIfUnused();
            } while (((int)rc & 0xFF) == (int)RC.BUSY && pBt.InTransaction == TRANS.NONE && btreeInvokeBusyHandler(pBt) != 0);
            if (rc == RC.OK)
            {
                if (this.inTrans == TRANS.NONE)
                {
                    pBt.Transactions++;
#if !SQLITE_OMIT_SHARED_CACHE
                    if (p.sharable)
                    {
                        Debug.Assert(p.locks.pBtree == p && p.locks.iTable == 1);
                        p.locks.eLock = LOCK.READ;
                        p.locks.pNext = pBt.pLock;
                        pBt.pLock = &p.Locks;
                    }
#endif
                }
                this.inTrans = (wrflag != 0 ? TRANS.WRITE : TRANS.READ);
                if (this.inTrans > pBt.InTransaction)
                    pBt.InTransaction = this.inTrans;
                if (wrflag != 0)
                {
                    var pPage1 = pBt.Page1;
#if !SQLITE_OMIT_SHARED_CACHE
                    Debug.Assert(!pBt.pWriter);
                    pBt.pWriter = p;
                    pBt.isExclusive = (u8)(wrflag > 1);
#endif
                    // If the db-size header field is incorrect (as it may be if an old client has been writing the database file), update it now. Doing
                    // this sooner rather than later means the database size can safely  re-read the database size from page 1 if a savepoint or transaction
                    // rollback occurs within the transaction.
                    if (pBt.Pages != ConvertEx.Get4(pPage1.Data, 28))
                    {
                        rc = Pager.sqlite3PagerWrite(pPage1.DbPage);
                        if (rc == RC.OK)
                            ConvertEx.Put4(pPage1.Data, 28, pBt.Pages);
                    }
                }
            }
        trans_begun:
            if (rc == RC.OK && wrflag != 0)
                // This call makes sure that the pager has the correct number of open savepoints. If the second parameter is greater than 0 and
                // the sub-journal is not already open, then it will be opened here.
                rc = pBt.Pager.sqlite3PagerOpenSavepoint(this.DB.nSavepoint);
            btreeIntegrity();
            sqlite3BtreeLeave();
            return rc;
        }

        internal RC sqlite3BtreeBeginStmt(int iStatement)
        {
            sqlite3BtreeEnter();
            Debug.Assert(this.inTrans == TRANS.WRITE);
            var pBt = this.Shared;
            Debug.Assert(!pBt.ReadOnly);
            Debug.Assert(iStatement > 0);
            Debug.Assert(iStatement > this.DB.nSavepoint);
            Debug.Assert(pBt.InTransaction == TRANS.WRITE);
            // At the pager level, a statement transaction is a savepoint with an index greater than all savepoints created explicitly using
            // SQL statements. It is illegal to open, release or rollback any such savepoints while the statement transaction savepoint is active.
            var rc = pBt.Pager.sqlite3PagerOpenSavepoint(iStatement);
            sqlite3BtreeLeave();
            return rc;
        }

        internal RC sqlite3BtreeSavepoint(SAVEPOINT op, int iSavepoint)
        {
            var rc = RC.OK;
            if (this != null && this.inTrans == TRANS.WRITE)
            {
                Debug.Assert(op == SAVEPOINT.RELEASE || op == SAVEPOINT.ROLLBACK);
                Debug.Assert(iSavepoint >= 0 || (iSavepoint == -1 && op == SAVEPOINT.ROLLBACK));
                sqlite3BtreeEnter();
                var pBt = this.Shared;
                rc = pBt.Pager.sqlite3PagerSavepoint(op, iSavepoint);
                if (rc == RC.OK)
                {
                    if (iSavepoint < 0 && pBt.InitiallyEmpty)
                        pBt.Pages = 0;
                    rc = pBt.newDatabase();
                    pBt.Pages = ConvertEx.Get4(pBt.Page1.Data, 28);
                    //The database size was written into the offset 28 of the header when the transaction started, so we know that the value at offset 28 is nonzero.
                    Debug.Assert(pBt.Pages > 0);
                }
                sqlite3BtreeLeave();
            }
            return rc;
        }

        internal RC btreeCursor(int iTable, bool wrFlag, KeyInfo pKeyInfo, BtreeCursor pCur)
        {
            Debug.Assert(sqlite3BtreeHoldsMutex());
            // The following Debug.Assert statements verify that if this is a sharable b-tree database, the connection is holding the required table locks,
            // and that no other connection has any open cursor that conflicts with this lock.
            Debug.Assert(hasSharedCacheTableLock(this, (uint)iTable, pKeyInfo != null ? 1 : 0, wrFlag ? 1 : 2));
            Debug.Assert(!wrFlag || !hasReadConflicts(this, (uint)iTable));
            // Assert that the caller has opened the required transaction.
            Debug.Assert(this.inTrans > TRANS.NONE);
            Debug.Assert(!wrFlag || this.inTrans == TRANS.WRITE);
            var pBt = this.Shared;                 // Shared b-tree handle
            Debug.Assert(pBt.Page1 != null && pBt.Page1.Data != null);
            if (Check.NEVER(wrFlag && pBt.ReadOnly))
                return RC.READONLY;
            if (iTable == 1 && pBt.btreePagecount() == 0)
                return RC.EMPTY;
            // Now that no other errors can occur, finish filling in the BtCursor variables and link the cursor into the BtShared list.
            pCur.RootID = (Pgno)iTable;
            pCur.PageID = -1;
            pCur.KeyInfo = pKeyInfo;
            pCur.Tree = this;
            pCur.Shared = pBt;
            pCur.Writeable = wrFlag;
            pCur.Next = pBt.Cursors;
            if (pCur.Next != null)
                pCur.Next.Prev = pCur;
            pBt.Cursors = pCur;
            pCur.State = CURSOR.INVALID;
            pCur._cachedRowID = 0;
            return RC.OK;
        }

        internal RC sqlite3BtreeCursor(int iTable, bool wrFlag, KeyInfo pKeyInfo, BtreeCursor pCur)
        {
            sqlite3BtreeEnter();
            var rc = btreeCursor(iTable, wrFlag, pKeyInfo, pCur);
            sqlite3BtreeLeave();
            return rc;
        }

        internal RC btreeCreateTable(ref int piTable, CREATETABLE createTabFlags)
        {
            var pBt = this.Shared;
            var pRoot = new MemPage();
            Pgno pgnoRoot = 0;
            int ptfFlags;          // Page-type flage for the root page of new table
            RC rc;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(pBt.InTransaction == TRANS.WRITE);
            Debug.Assert(!pBt.ReadOnly);
#if SQLITE_OMIT_AUTOVACUUM
            rc = allocateBtreePage(pBt, ref pRoot, ref pgnoRoot, 1, 0);
            if (rc != SQLITE.OK)
                return rc;
#else
            if (pBt.AutoVacuum)
            {
                Pgno pgnoMove = 0; // Move a page here to make room for the root-page
                var pPageMove = new MemPage(); // The page to move to.
                // Creating a new table may probably require moving an existing database to make room for the new tables root page. In case this page turns
                // out to be an overflow page, delete all overflow page-map caches held by open cursors.
                invalidateAllOverflowCache(pBt);
                // Read the value of meta[3] from the database to determine where the root page of the new table should go. meta[3] is the largest root-page
                // created so far, so the new root-page is (meta[3]+1).
                sqlite3BtreeGetMeta((int)META.LARGEST_ROOT_PAGE, ref pgnoRoot);
                pgnoRoot++;
                // The new root-page may not be allocated on a pointer-map page, or the PENDING_BYTE page.
                while (pgnoRoot == MemPage.PTRMAP_PAGENO(pBt, pgnoRoot) || pgnoRoot == MemPage.PENDING_BYTE_PAGE(pBt))
                    pgnoRoot++;
                Debug.Assert(pgnoRoot >= 3);
                // Allocate a page. The page that currently resides at pgnoRoot will be moved to the allocated page (unless the allocated page happens to reside at pgnoRoot).
                rc = pBt.allocateBtreePage(ref pPageMove, ref pgnoMove, pgnoRoot, 1);
                if (rc != RC.OK)
                    return rc;
                if (pgnoMove != pgnoRoot)
                {
                    // pgnoRoot is the page that will be used for the root-page of the new table (assuming an error did not occur). But we were
                    // allocated pgnoMove. If required (i.e. if it was not allocated by extending the file), the current page at position pgnoMove
                    // is already journaled.
                    PTRMAP eType = 0;
                    Pgno iPtrPage = 0;
                    pPageMove.releasePage();
                    // Move the page currently at pgnoRoot to pgnoMove.
                    rc = pBt.btreeGetPage(pgnoRoot, ref pRoot, 0);
                    if (rc != RC.OK)
                        return rc;
                    rc = pBt.ptrmapGet(pgnoRoot, ref eType, ref iPtrPage);
                    if (eType == PTRMAP.ROOTPAGE || eType == PTRMAP.FREEPAGE)
                        rc = SysEx.SQLITE_CORRUPT_BKPT();
                    if (rc != RC.OK)
                    {
                        pRoot.releasePage();
                        return rc;
                    }
                    Debug.Assert(eType != PTRMAP.ROOTPAGE);
                    Debug.Assert(eType != PTRMAP.FREEPAGE);
                    rc = MemPage.relocatePage(pBt, pRoot, eType, iPtrPage, pgnoMove, 0);
                    pRoot.releasePage();
                    // Obtain the page at pgnoRoot
                    if (rc != RC.OK)
                        return rc;
                    rc = pBt.btreeGetPage(pgnoRoot, ref pRoot, 0);
                    if (rc != RC.OK)
                        return rc;
                    rc = Pager.sqlite3PagerWrite(pRoot.DbPage);
                    if (rc != RC.OK)
                    {
                        pRoot.releasePage();
                        return rc;
                    }
                }
                else
                    pRoot = pPageMove;
                // Update the pointer-map and meta-data with the new root-page number.
                pBt.ptrmapPut(pgnoRoot, PTRMAP.ROOTPAGE, 0, ref rc);
                if (rc != RC.OK)
                {
                    pRoot.releasePage();
                    return rc;
                }
                // When the new root page was allocated, page 1 was made writable in order either to increase the database filesize, or to decrement the
                // freelist count.  Hence, the sqlite3BtreeUpdateMeta() call cannot fail.
                Debug.Assert(Pager.sqlite3PagerIswriteable(pBt.Page1.DbPage));
                rc = sqlite3BtreeUpdateMeta(4, pgnoRoot);
                if (Check.NEVER(rc != RC.OK))
                {
                    pRoot.releasePage();
                    return rc;
                }
            }
            else
            {
                rc = pBt.allocateBtreePage(ref pRoot, ref pgnoRoot, 1, 0);
                if (rc != RC.OK)
                    return rc;
            }
#endif
            Debug.Assert(Pager.sqlite3PagerIswriteable(pRoot.DbPage));
            ptfFlags = ((createTabFlags & CREATETABLE.INTKEY) != 0 ? PTF_INTKEY | PTF_LEAFDATA | PTF_LEAF : PTF_ZERODATA | PTF_LEAF);
            pRoot.zeroPage(ptfFlags);
            Pager.sqlite3PagerUnref(pRoot.DbPage);
            Debug.Assert((pBt.OpenFlags & OPEN.SINGLE) == 0 || pgnoRoot == 2);
            piTable = (int)pgnoRoot;
            return RC.OK;
        }

        internal RC sqlite3BtreeCreateTable(ref int piTable, CREATETABLE flags)
        {
            sqlite3BtreeEnter();
            var rc = btreeCreateTable(ref piTable, flags);
            sqlite3BtreeLeave();
            return rc;
        }

        internal RC sqlite3BtreeClearTable(int iTable, ref int pnChange)
        {
            var pBt = this.Shared;
            sqlite3BtreeEnter();
            Debug.Assert(this.inTrans == TRANS.WRITE);
            // Invalidate all incrblob cursors open on table iTable (assuming iTable is the root of a table b-tree - if it is not, the following call is a no-op).
            invalidateIncrblobCursors(this, 0, true);
            var rc = pBt.saveAllCursors((Pgno)iTable, null);
            if (rc == RC.OK)
                rc = pBt.clearDatabasePage((Pgno)iTable, 0, ref pnChange);
            sqlite3BtreeLeave();
            return rc;
        }

        internal RC btreeDropTable(Pgno iTable, ref int piMoved)
        {
            MemPage pPage = null;
            var pBt = this.Shared;
            Debug.Assert(sqlite3BtreeHoldsMutex());
            Debug.Assert(this.inTrans == TRANS.WRITE);
            // It is illegal to drop a table if any cursors are open on the database. This is because in auto-vacuum mode the backend may
            // need to move another root-page to fill a gap left by the deleted root page. If an open cursor was using this page a problem would occur.
            // This error is caught long before control reaches this point.
            if (Check.NEVER(pBt.Cursors))
            {
                sqlite3.sqlite3ConnectionBlocked(this.DB, pBt.Cursors.Tree.DB);
                return RC.LOCKED_SHAREDCACHE;
            }
            var rc = pBt.btreeGetPage((Pgno)iTable, ref pPage, 0);
            if (rc != RC.OK)
                return rc;
            var dummy0 = 0;
            rc = sqlite3BtreeClearTable((int)iTable, ref dummy0);
            if (rc != RC.OK)
            {
                pPage.releasePage();
                return rc;
            }
            piMoved = 0;
            if (iTable > 1)
            {
#if SQLITE_OMIT_AUTOVACUUM
freePage(pPage, ref rc);
releasePage(pPage);
#else
                if (pBt.AutoVacuum)
                {
                    Pgno maxRootPgno = 0;
                    sqlite3BtreeGetMeta((int)META.LARGEST_ROOT_PAGE, ref maxRootPgno);
                    if (iTable == maxRootPgno)
                    {
                        // If the table being dropped is the table with the largest root-page number in the database, put the root page on the free list.
                        pPage.freePage(ref rc);
                        pPage.releasePage();
                        if (rc != RC.OK)
                            return rc;
                    }
                    else
                    {
                        // The table being dropped does not have the largest root-page number in the database. So move the page that does into the
                        // gap left by the deleted root-page.
                        var pMove = new MemPage();
                        pPage.releasePage();
                        rc = pBt.btreeGetPage(maxRootPgno, ref pMove, 0);
                        if (rc != RC.OK)
                            return rc;
                        rc = MemPage.relocatePage(pBt, pMove, PTRMAP.ROOTPAGE, 0, iTable, 0);
                        pMove.releasePage();
                        if (rc != RC.OK)
                            return rc;
                        pMove = null;
                        rc = pBt.btreeGetPage(maxRootPgno, ref pMove, 0);
                        pMove.freePage(ref rc);
                        pMove.releasePage();
                        if (rc != RC.OK)
                            return rc;
                        piMoved = (int)maxRootPgno;
                    }
                    // Set the new 'max-root-page' value in the database header. This is the old value less one, less one more if that happens to
                    // be a root-page number, less one again if that is the PENDING_BYTE_PAGE.
                    maxRootPgno--;
                    while (maxRootPgno == MemPage.PENDING_BYTE_PAGE(pBt) || MemPage.PTRMAP_ISPAGE(pBt, maxRootPgno))
                        maxRootPgno--;
                    Debug.Assert(maxRootPgno != MemPage.PENDING_BYTE_PAGE(pBt));
                    rc = sqlite3BtreeUpdateMeta(4, maxRootPgno);
                }
                else
                {
                    pPage.freePage(ref rc);
                    pPage.releasePage();
                }
#endif
            }
            else
            {
                // If sqlite3BtreeDropTable was called on page 1. This really never should happen except in a corrupt database.
                pPage.zeroPage(PTF_INTKEY | PTF_LEAF);
                pPage.releasePage();
            }
            return rc;
        }

        internal RC sqlite3BtreeDropTable(int iTable, ref int piMoved)
        {
            sqlite3BtreeEnter();
            var rc = btreeDropTable((uint)iTable, ref piMoved);
            sqlite3BtreeLeave();
            return rc;
        }

        internal void sqlite3BtreeGetMeta(int idx, ref uint pMeta)
        {
            var pBt = this.Shared;
            sqlite3BtreeEnter();
            Debug.Assert(this.inTrans > TRANS.NONE);
            Debug.Assert(querySharedCacheTableLock(this, MASTER_ROOT, BtreeLock.LOCK.READ) == RC.OK);
            Debug.Assert(pBt.Page1 != null);
            Debug.Assert(idx >= 0 && idx <= 15);
            pMeta = ConvertEx.Get4(pBt.Page1.Data, 36 + idx * 4);
            // If auto-vacuum is disabled in this build and this is an auto-vacuum database, mark the database as read-only.
#if SQLITE_OMIT_AUTOVACUUM
            if (idx == BTREE_LARGEST_ROOT_PAGE && pMeta > 0) pBt.readOnly = 1;
#endif
            sqlite3BtreeLeave();
        }

        internal RC sqlite3BtreeUpdateMeta(int idx, uint iMeta)
        {
            var pBt = this.Shared;
            Debug.Assert(idx >= 1 && idx <= 15);
            sqlite3BtreeEnter();
            Debug.Assert(this.inTrans == TRANS.WRITE);
            Debug.Assert(pBt.Page1 != null);
            var pP1 = pBt.Page1.Data;
            var rc = Pager.sqlite3PagerWrite(pBt.Page1.DbPage);
            if (rc == RC.OK)
            {
                ConvertEx.Put4L(pP1, 36 + idx * 4, iMeta);
#if !SQLITE_OMIT_AUTOVACUUM
                if (idx == (int)META.INCR_VACUUM)
                {
                    Debug.Assert(pBt.AutoVacuum || iMeta == 0);
                    Debug.Assert(iMeta == 0 || iMeta == 1);
                    pBt.IncrVacuum = iMeta != 0;
                }
#endif
            }
            sqlite3BtreeLeave();
            return rc;
        }

#if !SQLITE_OMIT_WAL
static int sqlite3BtreeCheckpointBtree *p, int eMode, int *pnLog, int *pnCkpt){
int rc = SQLITE_OK;
if( p != null){
BtShared pBt = p.pBt;
sqlite3BtreeEnter(p);
if( pBt.inTransaction!=TRANS_NONE ){
rc = SQLITE_LOCKED;
}else{
rc = sqlite3PagerCheckpoint(pBt.pPager, eMode, pnLog, pnCkpt);
}
sqlite3BtreeLeave(p);
}
return rc;
}
#endif

        internal bool sqlite3BtreeIsInReadTrans()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            return (this.inTrans != TRANS.NONE);
        }

        internal bool sqlite3BtreeIsInBackup()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            return (this.nBackup != 0);
        }

        internal ISchema sqlite3BtreeSchema(int nBytes, Func<ISchema> xNew, Action<ISchema> xFree)
        {
            var pBt = this.Shared;
            sqlite3BtreeEnter();
            if (pBt.Schema == null && nBytes != 0)
            {
                pBt.Schema = xNew();
                pBt.xFreeSchema = xFree;
            }
            sqlite3BtreeLeave();
            return pBt.Schema;
        }

        internal RC sqlite3BtreeSchemaLocked()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            sqlite3BtreeEnter();
            var rc = querySharedCacheTableLock(this, MASTER_ROOT, BtreeLock.LOCK.READ);
            Debug.Assert(rc == RC.OK || rc == RC.LOCKED_SHAREDCACHE);
            sqlite3BtreeLeave();
            return rc;
        }

#if !SQLITE_OMIT_SHARED_CACHE
        /*
** Obtain a lock on the table whose root page is iTab.  The
** lock is a write lock if isWritelock is true or a read lock
** if it is false.
*/
        int sqlite3BtreeLockTable(Btree p, int iTab, byte isWriteLock)
        {
            int rc = SQLITE_OK;
            Debug.Assert(p.inTrans != TRANS_NONE);
            if (p.sharable)
            {
                u8 lockType = READ_LOCK + isWriteLock;
                Debug.Assert(READ_LOCK + 1 == WRITE_LOCK);
                Debug.Assert(isWriteLock == null || isWriteLock == 1);

                sqlite3BtreeEnter(p);
                rc = querySharedCacheTableLock(p, iTab, lockType);
                if (rc == SQLITE_OK)
                {
                    rc = setSharedCacheTableLock(p, iTab, lockType);
                }
                sqlite3BtreeLeave(p);
            }
            return rc;
        }
#endif

        internal RC sqlite3BtreeSetVersion(int iVersion)
        {
            var pBt = this.Shared;
            Debug.Assert(this.inTrans == TRANS.NONE);
            Debug.Assert(iVersion == 1 || iVersion == 2);
            // If setting the version fields to 1, do not automatically open the WAL connection, even if the version fields are currently set to 2.
            pBt.DoNotUseWal = iVersion == 1;
            var rc = sqlite3BtreeBeginTrans(0);
            if (rc == RC.OK)
            {
                var aData = pBt.Page1.Data;
                if (aData[18] != (byte)iVersion || aData[19] != (byte)iVersion)
                {
                    rc = sqlite3BtreeBeginTrans(2);
                    if (rc == RC.OK)
                    {
                        rc = Pager.sqlite3PagerWrite(pBt.Page1.DbPage);
                        if (rc == RC.OK)
                        {
                            aData[18] = (byte)iVersion;
                            aData[19] = (byte)iVersion;
                        }
                    }
                }
            }
            pBt.DoNotUseWal = false;
            return rc;
        }
    }
}
