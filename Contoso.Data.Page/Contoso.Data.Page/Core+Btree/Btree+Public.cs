using System;
using System.Diagnostics;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using DbPage = Contoso.Core.PgHdr;
using Pgno = System.UInt32;
using PTRMAP = Contoso.Core.MemPage.PTRMAP;
using SAVEPOINT = Contoso.Core.Pager.SAVEPOINT;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;
using MUTEX = Contoso.Core.MutexEx.MUTEX;
using LOCK = Contoso.Core.Btree.BtreeLock.LOCK;

namespace Contoso.Core
{
    public partial class Btree
    {
        // was:sqlite3BtreeOpen
        public static RC Open(VirtualFileSystem pVfs, string zFilename, sqlite3 db, ref Btree rTree, OPEN flags, VFSOPEN vfsFlags)
        {
            Btree p;                      // Handle to return   
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
            p.InTransaction = TRANS.NONE;
            p.DB = db;
#if !SQLITE_OMIT_SHARED_CACHE
            p.Locks.Tree = p;
            p.Locks.TableID = 1;
#endif
            BtShared shared = null;          // Shared part of btree structure
            sqlite3_mutex mutexOpen = null;  // Prevents a race condition.
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
            // If this Btree is a candidate for shared cache, try to find an existing BtShared object that we can share with
            if (!isMemdb && !isTempDb)
            {
                if ((vfsFlags & VFSOPEN.SHAREDCACHE) != 0)
                {
                    p.Sharable = true;
                    string zPathname;
                    rc = pVfs.xFullPathname(zFilename, out zPathname);
                    mutexOpen = MutexEx.sqlite3MutexAlloc(MUTEX.STATIC_OPEN);
                    MutexEx.sqlite3_mutex_enter(mutexOpen);
                    var mutexShared = MutexEx.sqlite3MutexAlloc(MUTEX.STATIC_MASTER);
                    MutexEx.sqlite3_mutex_enter(mutexShared);
                    for (shared = SysEx.getGLOBAL<BtShared>(s_sqlite3SharedCacheList); shared != null; shared = shared.Next)
                    {
                        Debug.Assert(shared.nRef > 0);
                        if (string.Equals(zPathname, shared.Pager.sqlite3PagerFilename()) && shared.Pager.sqlite3PagerVfs() == pVfs)
                        {
                            for (var iDb = db.DBs - 1; iDb >= 0; iDb--)
                            {
                                var existingTree = db.AllocDBs[iDb].Tree;
                                if (existingTree != null && existingTree.Shared == shared)
                                {
                                    MutexEx.sqlite3_mutex_leave(mutexShared);
                                    MutexEx.sqlite3_mutex_leave(mutexOpen);
                                    p = null;
                                    return RC.CONSTRAINT;
                                }
                            }
                            p.Shared = shared;
                            shared.nRef++;
                            break;
                        }
                    }
                    MutexEx.sqlite3_mutex_leave(mutexShared);
                }
#if DEBUG
                else
                    // In debug mode, we mark all persistent databases as sharable even when they are not.  This exercises the locking code and
                    // gives more opportunity for asserts(sqlite3_mutex_held()) statements to find locking problems.
                    p.Sharable = true;
#endif
            }
#endif
            if (shared == null)
            {
                // The following asserts make sure that structures used by the btree are the right size.  This is to guard against size changes that result
                // when compiling on a different architecture.
                Debug.Assert(sizeof(long) == 8 || sizeof(long) == 4);
                Debug.Assert(sizeof(ulong) == 8 || sizeof(ulong) == 4);
                Debug.Assert(sizeof(uint) == 4);
                Debug.Assert(sizeof(ushort) == 2);
                Debug.Assert(sizeof(Pgno) == 4);
                shared = new BtShared();
                rc = Pager.sqlite3PagerOpen(pVfs, out shared.Pager, zFilename, EXTRA_SIZE, (Pager.PAGEROPEN)flags, vfsFlags, pageReinit);
                if (rc == RC.OK)
                    rc = shared.Pager.sqlite3PagerReadFileheader(zDbHeader.Length, zDbHeader);
                if (rc != RC.OK)
                    goto btree_open_out;
                shared.OpenFlags = flags;
                shared.DB = db;
                shared.Pager.sqlite3PagerSetBusyhandler(btreeInvokeBusyHandler, shared);
                p.Shared = shared;
                shared.Cursors = null;
                shared.Page1 = null;
                shared.ReadOnly = shared.Pager.sqlite3PagerIsreadonly();
#if SQLITE_SECURE_DELETE
pBt.secureDelete = true;
#endif
                shared.PageSize = (uint)((zDbHeader[16] << 8) | (zDbHeader[17] << 16)); if (shared.PageSize < 512 || shared.PageSize > Pager.SQLITE_MAX_PAGE_SIZE || ((shared.PageSize - 1) & shared.PageSize) != 0)
                {
                    shared.PageSize = 0;
#if !SQLITE_OMIT_AUTOVACUUM
                    // If the magic name ":memory:" will create an in-memory database, then leave the autoVacuum mode at 0 (do not auto-vacuum), even if
                    // SQLITE_DEFAULT_AUTOVACUUM is true. On the other hand, if SQLITE_OMIT_MEMORYDB has been defined, then ":memory:" is just a
                    // regular file-name. In this case the auto-vacuum applies as per normal.
                    if (zFilename != string.Empty && !isMemdb)
                    {
                        shared.AutoVacuum = (AUTOVACUUM.DEFAULT != AUTOVACUUM.NONE);
                        shared.IncrVacuum = (AUTOVACUUM.DEFAULT == AUTOVACUUM.INCR);
                    }
#endif
                    nReserve = 0;
                }
                else
                {
                    nReserve = zDbHeader[20];
                    shared.PageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
                    shared.AutoVacuum = ConvertEx.Get4(zDbHeader, 36 + 4 * 4) != 0;
                    shared.IncrVacuum = ConvertEx.Get4(zDbHeader, 36 + 7 * 4) != 0;
#endif
                }
                rc = shared.Pager.sqlite3PagerSetPagesize(ref shared.PageSize, nReserve);
                if (rc != RC.OK)
                    goto btree_open_out;
                shared.UsableSize = (ushort)(shared.PageSize - nReserve);
                Debug.Assert((shared.PageSize & 7) == 0);  // 8-byte alignment of pageSize
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
                // Add the new BtShared object to the linked list sharable BtShareds.
                if (p.Sharable)
                {
                    sqlite3_mutex mutexShared;
                    shared.nRef = 1;
                    mutexShared = MutexEx.sqlite3MutexAlloc(MUTEX.STATIC_MASTER);
                    if (MutexEx.SQLITE_THREADSAFE && MutexEx.WantsCoreMutex)
                        shared.Mutex = MutexEx.sqlite3MutexAlloc(MUTEX.FAST);
                    MutexEx.sqlite3_mutex_enter(mutexShared);
                    shared.Next = SysEx.getGLOBAL<BtShared>(s_sqlite3SharedCacheList);
                    SysEx.setGLOBAL<BtShared>(s_sqlite3SharedCacheList, shared);
                    MutexEx.sqlite3_mutex_leave(mutexShared);
                }
#endif
            }
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
            // If the new Btree uses a sharable pBtShared, then link the new Btree into the list of all sharable Btrees for the same connection.
            // The list is kept in ascending order by pBt address.
            Btree existingTree2;
            if (p.Sharable)
                for (var i = 0; i < db.DBs; i++)
                    if ((existingTree2 = db.AllocDBs[i].Tree) != null && existingTree2.Sharable)
                    {
                        while (existingTree2.Prev != null) { existingTree2 = existingTree2.Prev; }
                        if (p.Shared.Version < existingTree2.Shared.Version)
                        {
                            p.Next = existingTree2;
                            p.Prev = null;
                            existingTree2.Prev = p;
                        }
                        else
                        {
                            while (existingTree2.Next != null && existingTree2.Next.Shared.Version < p.Shared.Version)
                                existingTree2 = existingTree2.Next;
                            p.Next = existingTree2.Next;
                            p.Prev = existingTree2;
                            if (p.Next != null)
                                p.Next.Prev = p;
                            existingTree2.Next = p;
                        }
                        break;
                    }
#endif
            rTree = p;
        //
        btree_open_out:
            if (rc != RC.OK)
            {
                if (shared != null && shared.Pager != null)
                    shared.Pager.sqlite3PagerClose();
                shared = null;
                p = null;
                rTree = null;
            }
            else
            {
                // If the B-Tree was successfully opened, set the pager-cache size to the default value. Except, when opening on an existing shared pager-cache,
                // do not change the pager-cache size.
                if (p.GetSchema(0, null, null) == null)
                    p.Shared.Pager.sqlite3PagerSetCachesize(SQLITE_DEFAULT_CACHE_SIZE);
            }
            if (mutexOpen != null)
            {
                Debug.Assert(MutexEx.Held(mutexOpen));
                MutexEx.sqlite3_mutex_leave(mutexOpen);
            }
            return rc;
        }

        // was:sqlite3BtreeClose
        public static RC Close(ref Btree p)
        {
            // Close all cursors opened via this handle.
            Debug.Assert(MutexEx.Held(p.DB.Mutex));
            p.sqlite3BtreeEnter();
            var shared = p.Shared;
            var cursor = shared.Cursors;
            while (cursor != null)
            {
                var lastCursor = cursor;
                cursor = cursor.Next;
                if (lastCursor.Tree == p)
                    lastCursor.Close();
            }
            // Rollback any active transaction and free the handle structure. The call to sqlite3BtreeRollback() drops any table-locks held by this handle.
            p.Rollback();
            p.sqlite3BtreeLeave();
            // If there are still other outstanding references to the shared-btree structure, return now. The remainder of this procedure cleans up the shared-btree.
            Debug.Assert(p.WantToLock == 0 && !p.Locked);
            if (!p.Sharable || shared.removeFromSharingList())
            {
                // The pBt is no longer on the sharing list, so we can access it without having to hold the mutex.
                // Clean out and delete the BtShared object.
                Debug.Assert(shared.Cursors == null);
                shared.Pager.sqlite3PagerClose();
                if (shared.xFreeSchema != null && shared.Schema != null)
                    shared.xFreeSchema(shared.Schema);
                shared.Schema = null;// sqlite3DbFree(0, pBt->pSchema);
                //freeTempSpace(pBt);
                shared = null; //sqlite3_free(ref pBt);
            }
#if !SQLITE_OMIT_SHARED_CACHE
            Debug.Assert(p.WantToLock == 0);
            Debug.Assert(p.Locked == false);
            if (p.Prev != null) p.Prev.Next = p.Next;
            if (p.Next != null) p.Next.Prev = p.Prev;
#endif
            return RC.OK;
        }

        // was:sqlite3BtreeSyncDisabled
        public int SyncDisabled()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            sqlite3BtreeEnter();
            var pBt = this.Shared;
            Debug.Assert(pBt != null && pBt.Pager != null);
            var rc = (pBt.Pager.sqlite3PagerNosync() ? 1 : 0);
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeBeginTrans
        public RC BeginTrans(byte wrflag)
        {
            var shared = this.Shared;
            var rc = RC.OK;
            sqlite3BtreeEnter();
            btreeIntegrity();
            // If the btree is already in a write-transaction, or it is already in a read-transaction and a read-transaction is requested, this is a no-op.
            if (this.InTransaction == TRANS.WRITE || (this.InTransaction == TRANS.READ && wrflag == 0))
                goto trans_begun;
            // Write transactions are not possible on a read-only database
            if (shared.ReadOnly && wrflag != 0)
            {
                rc = RC.READONLY;
                goto trans_begun;
            }
#if !SQLITE_OMIT_SHARED_CACHE
            // If another database handle has already opened a write transaction on this shared-btree structure and a second write transaction is
            // requested, return SQLITE_LOCKED.
            sqlite3 sharedDB = null;
            if ((wrflag != 0 && shared.InTransaction == TRANS.WRITE) || shared.IsPending)
                sharedDB = shared.Writer.DB;
            else if (wrflag > 1)
                for (var @lock = shared.Locks; @lock != null; @lock = @lock.Next)
                    if (@lock.Tree != this)
                    {
                        sharedDB = @lock.Tree.DB;
                        break;
                    }
            if (sharedDB != null)
            {
                sqlite3.sqlite3ConnectionBlocked(this.DB, sharedDB);
                rc = RC.LOCKED_SHAREDCACHE;
                goto trans_begun;
            }
#endif
            // Any read-only or read-write transaction implies a read-lock on page 1. So if some other shared-cache client already has a write-lock
            // on page 1, the transaction cannot be opened. */
            rc = querySharedCacheTableLock(MASTER_ROOT, LOCK.READ);
            if (rc != RC.OK)
                goto trans_begun;
            shared.InitiallyEmpty = shared.Pages == 0;
            do
            {
                // Call lockBtree() until either pBt.pPage1 is populated or lockBtree() returns something other than SQLITE_OK. lockBtree()
                // may return SQLITE_OK but leave pBt.pPage1 set to 0 if after reading page 1 it discovers that the page-size of the database
                // file is not pBt.pageSize. In this case lockBtree() will update pBt.pageSize to the page-size of the file on disk.
                while (shared.Page1 == null && (rc = shared.lockBtree()) == RC.OK) ;
                if (rc == RC.OK && wrflag != 0)
                {
                    if (shared.ReadOnly)
                        rc = RC.READONLY;
                    else
                    {
                        rc = shared.Pager.sqlite3PagerBegin(wrflag > 1, this.DB.sqlite3TempInMemory() ? 1 : 0);
                        if (rc == RC.OK)
                            rc = shared.newDatabase();
                    }
                }
                if (rc != RC.OK)
                    shared.unlockBtreeIfUnused();
            } while (((int)rc & 0xFF) == (int)RC.BUSY && shared.InTransaction == TRANS.NONE && btreeInvokeBusyHandler(shared) != 0);
            if (rc == RC.OK)
            {
                if (this.InTransaction == TRANS.NONE)
                {
                    shared.Transactions++;
#if !SQLITE_OMIT_SHARED_CACHE
                    if (Sharable)
                    {
                        Debug.Assert(Locks.Tree == this && Locks.TableID == 1);
                        Locks.Lock = LOCK.READ;
                        Locks.Next = shared.Locks;
                        shared.Locks = Locks;
                    }
#endif
                }
                this.InTransaction = (wrflag != 0 ? TRANS.WRITE : TRANS.READ);
                if (this.InTransaction > shared.InTransaction)
                    shared.InTransaction = this.InTransaction;
                if (wrflag != 0)
                {
                    var pPage1 = shared.Page1;
#if !SQLITE_OMIT_SHARED_CACHE
                    Debug.Assert(shared.Writer == null);
                    shared.Writer = this;
                    shared.IsExclusive = (wrflag > 1);
#endif
                    // If the db-size header field is incorrect (as it may be if an old client has been writing the database file), update it now. Doing
                    // this sooner rather than later means the database size can safely  re-read the database size from page 1 if a savepoint or transaction
                    // rollback occurs within the transaction.
                    if (shared.Pages != ConvertEx.Get4(pPage1.Data, 28))
                    {
                        rc = Pager.sqlite3PagerWrite(pPage1.DbPage);
                        if (rc == RC.OK)
                            ConvertEx.Put4(pPage1.Data, 28, shared.Pages);
                    }
                }
            }
        trans_begun:
            if (rc == RC.OK && wrflag != 0)
                // This call makes sure that the pager has the correct number of open savepoints. If the second parameter is greater than 0 and
                // the sub-journal is not already open, then it will be opened here.
                rc = shared.Pager.sqlite3PagerOpenSavepoint(this.DB.nSavepoint);
            btreeIntegrity();
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeBeginStmt
        public RC BeginStmt(int iStatement)
        {
            sqlite3BtreeEnter();
            Debug.Assert(this.InTransaction == TRANS.WRITE);
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

        // was:sqlite3BtreeSavepoint
        public RC Savepoint(SAVEPOINT op, int iSavepoint)
        {
            var rc = RC.OK;
            if (this != null && this.InTransaction == TRANS.WRITE)
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

        // was:sqlite3BtreeCursor
        public RC Cursor(int iTable, bool wrFlag, KeyInfo pKeyInfo, BtreeCursor pCur)
        {
            sqlite3BtreeEnter();
            var rc = btreeCursor(iTable, wrFlag, pKeyInfo, pCur);
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeCreateTable
        public RC CreateTable(ref int piTable, CREATETABLE flags)
        {
            sqlite3BtreeEnter();
            var rc = btreeCreateTable(ref piTable, flags);
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeClearTable
        public RC ClearTable(int iTable, ref int pnChange)
        {
            var pBt = this.Shared;
            sqlite3BtreeEnter();
            Debug.Assert(this.InTransaction == TRANS.WRITE);
            // Invalidate all incrblob cursors open on table iTable (assuming iTable is the root of a table b-tree - if it is not, the following call is a no-op).
            invalidateIncrblobCursors(this, 0, true);
            var rc = pBt.saveAllCursors((Pgno)iTable, null);
            if (rc == RC.OK)
                rc = pBt.clearDatabasePage((Pgno)iTable, 0, ref pnChange);
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeDropTable
        public RC DropTable(int iTable, ref int piMoved)
        {
            sqlite3BtreeEnter();
            var rc = btreeDropTable((uint)iTable, ref piMoved);
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeGetMeta
        public void GetMeta(int idx, ref uint pMeta)
        {
            var pBt = this.Shared;
            sqlite3BtreeEnter();
            Debug.Assert(this.InTransaction > TRANS.NONE);
            Debug.Assert(querySharedCacheTableLock(MASTER_ROOT, BtreeLock.LOCK.READ) == RC.OK);
            Debug.Assert(pBt.Page1 != null);
            Debug.Assert(idx >= 0 && idx <= 15);
            pMeta = ConvertEx.Get4(pBt.Page1.Data, 36 + idx * 4);
            // If auto-vacuum is disabled in this build and this is an auto-vacuum database, mark the database as read-only.
#if SQLITE_OMIT_AUTOVACUUM
            if (idx == BTREE_LARGEST_ROOT_PAGE && pMeta > 0) pBt.readOnly = 1;
#endif
            sqlite3BtreeLeave();
        }

        // was:sqlite3BtreeUpdateMeta
        public RC SetMeta(int idx, uint iMeta)
        {
            var pBt = this.Shared;
            Debug.Assert(idx >= 1 && idx <= 15);
            sqlite3BtreeEnter();
            Debug.Assert(this.InTransaction == TRANS.WRITE);
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

        // was:sqlite3BtreeSchema
        public ISchema GetSchema(int nBytes, Func<ISchema> xNew, Action<ISchema> xFree)
        {
            var shared = this.Shared;
            sqlite3BtreeEnter();
            if (shared.Schema == null && nBytes != 0)
            {
                shared.Schema = xNew();
                shared.xFreeSchema = xFree;
            }
            sqlite3BtreeLeave();
            return shared.Schema;
        }

        // was:sqlite3BtreeSchemaLocked
        public RC SchemaLocked()
        {
            Debug.Assert(MutexEx.Held(this.DB.Mutex));
            sqlite3BtreeEnter();
            var rc = querySharedCacheTableLock(MASTER_ROOT, LOCK.READ);
            Debug.Assert(rc == RC.OK || rc == RC.LOCKED_SHAREDCACHE);
            sqlite3BtreeLeave();
            return rc;
        }

#if !SQLITE_OMIT_SHARED_CACHE
        // was:sqlite3BtreeLockTable
        public RC LockTable(Pgno tableID, bool isWriteLock)
        {
            var rc = RC.OK;
            Debug.Assert(InTransaction != TRANS.NONE);
            if (Sharable)
            {
                var lockType = (isWriteLock ? LOCK.READ : LOCK.WRITE);
                sqlite3BtreeEnter();
                rc = querySharedCacheTableLock(tableID, lockType);
                if (rc == RC.OK)
                    rc = setSharedCacheTableLock(tableID, lockType);
                sqlite3BtreeLeave();
            }
            return rc;
        }
#endif

        // was:sqlite3BtreeCommitPhaseOne
        public RC CommitPhaseOne(string zMaster)
        {
            var rc = RC.OK;
            if (this.InTransaction == TRANS.WRITE)
            {
                var shared = this.Shared;
                sqlite3BtreeEnter();
#if !SQLITE_OMIT_AUTOVACUUM
                if (shared.AutoVacuum)
                {
                    rc = MemPage.autoVacuumCommit(shared);
                    if (rc != RC.OK)
                    {
                        sqlite3BtreeLeave();
                        return rc;
                    }
                }
#endif
                rc = shared.Pager.sqlite3PagerCommitPhaseOne(zMaster, false);
                sqlite3BtreeLeave();
            }
            return rc;
        }

        // was:sqlite3BtreeCommitPhaseTwo
        public RC CommitPhaseTwo(int bCleanup)
        {
            if (this.InTransaction == TRANS.NONE)
                return RC.OK;
            sqlite3BtreeEnter();
            btreeIntegrity();
            // If the handle has a write-transaction open, commit the shared-btrees transaction and set the shared state to TRANS_READ.
            if (this.InTransaction == TRANS.WRITE)
            {
                var shared = this.Shared;
                Debug.Assert(shared.InTransaction == TRANS.WRITE);
                Debug.Assert(shared.Transactions > 0);
                var rc = shared.Pager.sqlite3PagerCommitPhaseTwo();
                if (rc != RC.OK && bCleanup == 0)
                {
                    sqlite3BtreeLeave();
                    return rc;
                }
                shared.InTransaction = TRANS.READ;
            }
            btreeEndTransaction();
            sqlite3BtreeLeave();
            return RC.OK;
        }

        // was:sqlite3BtreeCommit
        public RC Commit()
        {
            sqlite3BtreeEnter();
            var rc = CommitPhaseOne(null);
            if (rc == RC.OK)
                rc = CommitPhaseTwo(0);
            sqlite3BtreeLeave();
            return rc;
        }

        // was:sqlite3BtreeTripAllCursors
        public void TripAllCursors(RC errCode)
        {
            sqlite3BtreeEnter();
            for (var cursor = this.Shared.Cursors; cursor != null; cursor = cursor.Next)
            {
                cursor.Clear();
                cursor.State = CURSOR.FAULT;
                cursor.SkipNext = errCode;
                for (var i = 0; i <= cursor.PageID; i++)
                {
                    cursor.Pages[i].releasePage();
                    cursor.Pages[i] = null;
                }
            }
            sqlite3BtreeLeave();
        }

        // was:sqlite3BtreeRollback
        public RC Rollback()
        {
            sqlite3BtreeEnter();
            var shared = this.Shared;
            var rc = shared.saveAllCursors(0, null);
#if !SQLITE_OMIT_SHARED_CACHE
            if (rc != RC.OK)
                // This is a horrible situation. An IO or malloc() error occurred whilst trying to save cursor positions. If this is an automatic rollback (as
                // the result of a constraint, malloc() failure or IO error) then the cache may be internally inconsistent (not contain valid trees) so
                // we cannot simply return the error to the caller. Instead, abort all queries that may be using any of the cursors that failed to save.
                TripAllCursors(rc);
#endif
            btreeIntegrity();
            if (this.InTransaction == TRANS.WRITE)
            {
                Debug.Assert(shared.InTransaction == TRANS.WRITE);
                var rc2 = shared.Pager.sqlite3PagerRollback();
                if (rc2 != RC.OK)
                    rc = rc2;
                // The rollback may have destroyed the pPage1.aData value.  So call btreeGetPage() on page 1 again to make
                // sure pPage1.aData is set correctly.
                var pPage1 = new MemPage();
                if (shared.btreeGetPage(1, ref pPage1, 0) == RC.OK)
                {
                    Pgno nPage = ConvertEx.Get4(pPage1.Data, 28);
                    if (nPage == 0)
                        shared.Pager.sqlite3PagerPagecount(out nPage);
                    shared.Pages = nPage;
                    pPage1.releasePage();
                }
                Debug.Assert(shared.countWriteCursors() == 0);
                shared.InTransaction = TRANS.READ;
            }
            btreeEndTransaction();
            sqlite3BtreeLeave();
            return rc;
        }
    }
}
