using Pgno = System.UInt32;
using DbPage = Contoso.Core.PgHdr;
using System;
using System.Text;
using Contoso.Sys;
using System.Diagnostics;
using CURSOR = Contoso.Core.BtCursor.CURSOR;
using VFSOPEN = Contoso.Sys.VirtualFileSystem.OPEN;
namespace Contoso.Core
{
    public partial class Btree
    {
        internal static byte[] zMagicHeader = Encoding.UTF8.GetBytes(SQLITE_FILE_HEADER);

#if TRACE
        internal static bool sqlite3BtreeTrace = true;  // True to enable tracing
        internal static void TRACE(string x, params object[] args) { if (sqlite3BtreeTrace)Console.WriteLine(string.Format(x, args); }
#else
        internal static void TRACE(string x, params object[] args) { }
#endif

#if !SQLITE_OMIT_SHARED_CACHE
        int sqlite3_enable_shared_cache(int enable) { sqlite3GlobalConfig.sharedCacheEnabled = enable; return SQLITE_OK; }
#if DEBUG
static int hasSharedCacheTableLock(
Btree pBtree,         /* Handle that must hold lock */
Pgno iRoot,            /* Root page of b-tree */
int isIndex,           /* True if iRoot is the root of an index b-tree */
int eLockType          /* Required lock type (READ_LOCK or WRITE_LOCK) */
){
Schema pSchema = (Schema *)pBtree.pBt.pSchema;
Pgno iTab = 0;
BtLock pLock;

// If this database is not shareable, or if the client is reading and has the read-uncommitted flag set, then no lock is required.  Return true immediately.
if( (pBtree.sharable==null) || (eLockType==READ_LOCK && (pBtree.db.flags & SQLITE_ReadUncommitted)) ){
return 1;
}
// If the client is reading  or writing an index and the schema is not loaded, then it is too difficult to actually check to see if
// the correct locks are held.  So do not bother - just return true. This case does not come up very often anyhow.
if( isIndex && (!pSchema || (pSchema->flags&DB_SchemaLoaded)==0) ){
return 1;
}

// Figure out the root-page that the lock should be held on. For table b-trees, this is just the root page of the b-tree being read or
// written. For index b-trees, it is the root page of the associated table.
if( isIndex ){
HashElem p;
for(p=sqliteHashFirst(pSchema.idxHash); p!=null; p=sqliteHashNext(p)){
Index pIdx = (Index *)sqliteHashData(p);
if( pIdx.tnum==(int)iRoot ){
iTab = pIdx.pTable.tnum;
}
}
}else{
iTab = iRoot;
}
// Search for the required lock. Either a write-lock on root-page iTab, a write-lock on the schema table, or (if the client is reading) a
// read-lock on iTab will suffice. Return 1 if any of these are found.
for(pLock=pBtree.pBt.pLock; pLock; pLock=pLock.pNext){
if( pLock.pBtree==pBtree
&& (pLock.iTable==iTab || (pLock.eLock==WRITE_LOCK && pLock.iTable==1))
&& pLock.eLock>=eLockType
){
return 1;
}
}
// Failed to find the required lock.
return 0;
}
static int hasReadConflicts(Btree pBtree, Pgno iRoot){
BtCursor p;
for(p=pBtree.pBt.pCursor; p!=null; p=p.pNext){
if( p.pgnoRoot==iRoot
&& p.pBtree!=pBtree
&& 0==(p.pBtree.db.flags & SQLITE_ReadUncommitted)
){
return 1;
}
}
return 0;
}
#endif

static int querySharedCacheTableLock(Btree p, Pgno iTab, u8 eLock){
BtShared pBt = p.pBt;
BtLock pIter;
Debug.Assert( sqlite3BtreeHoldsMutex(p) );
Debug.Assert( eLock==READ_LOCK || eLock==WRITE_LOCK );
Debug.Assert( p.db!=null );
Debug.Assert( !(p.db.flags&SQLITE_ReadUncommitted)||eLock==WRITE_LOCK||iTab==1 );
// If requesting a write-lock, then the Btree must have an open write transaction on this file. And, obviously, for this to be so there
// must be an open write transaction on the file itself.
Debug.Assert( eLock==READ_LOCK || (p==pBt.pWriter && p.inTrans==TRANS_WRITE) );
Debug.Assert( eLock==READ_LOCK || pBt.inTransaction==TRANS_WRITE );
// This routine is a no-op if the shared-cache is not enabled 
if( !p.sharable ){
return SQLITE_OK;
}
// If some other connection is holding an exclusive lock, the requested lock may not be obtained.
if( pBt.pWriter!=p && pBt.isExclusive ){
sqlite3ConnectionBlocked(p.db, pBt.pWriter.db);
return SQLITE_LOCKED_SHAREDCACHE;
}
for(pIter=pBt.pLock; pIter; pIter=pIter.pNext){
// The condition (pIter.eLock!=eLock) in the following if(...) statement is a simplification of:
//   (eLock==WRITE_LOCK || pIter.eLock==WRITE_LOCK)
// since we know that if eLock==WRITE_LOCK, then no other connection may hold a WRITE_LOCK on any table in this file (since there can
// only be a single writer).
Debug.Assert( pIter.eLock==READ_LOCK || pIter.eLock==WRITE_LOCK );
Debug.Assert( eLock==READ_LOCK || pIter.pBtree==p || pIter.eLock==READ_LOCK);
if( pIter.pBtree!=p && pIter.iTable==iTab && pIter.eLock!=eLock ){
sqlite3ConnectionBlocked(p.db, pIter.pBtree.db);
if( eLock==WRITE_LOCK ){
Debug.Assert( p==pBt.pWriter );
pBt.isPending = 1;
}
return SQLITE_LOCKED_SHAREDCACHE;
}
}
return SQLITE_OK;
}

static int setSharedCacheTableLock(Btree p, Pgno iTable, u8 eLock){
BtShared pBt = p.pBt;
BtLock pLock = 0;
BtLock pIter;
Debug.Assert( sqlite3BtreeHoldsMutex(p) );
Debug.Assert( eLock==READ_LOCK || eLock==WRITE_LOCK );
Debug.Assert( p.db!=null );
// A connection with the read-uncommitted flag set will never try to obtain a read-lock using this function. The only read-lock obtained
// by a connection in read-uncommitted mode is on the sqlite_master table, and that lock is obtained in BtreeBeginTrans().
Debug.Assert( 0==(p.db.flags&SQLITE_ReadUncommitted) || eLock==WRITE_LOCK );
// This function should only be called on a sharable b-tree after it has been determined that no other b-tree holds a conflicting lock.
Debug.Assert( p.sharable );
Debug.Assert( SQLITE_OK==querySharedCacheTableLock(p, iTable, eLock) );
// First search the list for an existing lock on this table.
for(pIter=pBt.pLock; pIter; pIter=pIter.pNext){
if( pIter.iTable==iTable && pIter.pBtree==p ){
pLock = pIter;
break;
}
}
// If the above search did not find a BtLock struct associating Btree p with table iTable, allocate one and link it into the list.
if( !pLock ){
pLock = (BtLock *)sqlite3MallocZero(sizeof(BtLock));
if( !pLock ){
return SQLITE_NOMEM;
}
pLock.iTable = iTable;
pLock.pBtree = p;
pLock.pNext = pBt.pLock;
pBt.pLock = pLock;
}
// Set the BtLock.eLock variable to the maximum of the current lock and the requested lock. This means if a write-lock was already held
// and a read-lock requested, we don't incorrectly downgrade the lock.
Debug.Assert( WRITE_LOCK>READ_LOCK );
if( eLock>pLock.eLock ){
pLock.eLock = eLock;
}
return SQLITE_OK;
}
static void clearAllSharedCacheTableLocks(Btree p){
BtShared pBt = p.pBt;
BtLock **ppIter = &pBt.pLock;
Debug.Assert( sqlite3BtreeHoldsMutex(p) );
Debug.Assert( p.sharable || 0==*ppIter );
Debug.Assert( p.inTrans>0 );
while( ppIter ){
BtLock pLock = ppIter;
Debug.Assert( pBt.isExclusive==null || pBt.pWriter==pLock.pBtree );
Debug.Assert( pLock.pBtree.inTrans>=pLock.eLock );
if( pLock.pBtree==p ){
ppIter = pLock.pNext;
Debug.Assert( pLock.iTable!=1 || pLock==&p.lock );
if( pLock.iTable!=1 ){
pLock=null;//sqlite3_free(ref pLock);
}
}else{
ppIter = &pLock.pNext;
}
}

Debug.Assert( pBt.isPending==null || pBt.pWriter );
if( pBt.pWriter==p ){
pBt.pWriter = 0;
pBt.isExclusive = 0;
pBt.isPending = 0;
}else if( pBt.nTransaction==2 ){
/* This function is called when Btree p is concluding its 
** transaction. If there currently exists a writer, and p is not
** that writer, then the number of locks held by connections other
** than the writer must be about to drop to zero. In this case
** set the isPending flag to 0.
**
** If there is not currently a writer, then BtShared.isPending must
** be zero already. So this next line is harmless in that case.
*/
pBt.isPending = 0;
}
}

static void downgradeAllSharedCacheTableLocks(Btree p){
BtShared pBt = p.pBt;
if( pBt.pWriter==p ){
BtLock pLock;
pBt.pWriter = 0;
pBt.isExclusive = 0;
pBt.isPending = 0;
for(pLock=pBt.pLock; pLock; pLock=pLock.pNext){
Debug.Assert( pLock.eLock==READ_LOCK || pLock.pBtree==p );
pLock.eLock = READ_LOCK;
}
}
}
#else
        static SQLITE querySharedCacheTableLock(Btree p, Pgno iTab, byte eLock) { return SQLITE.OK; }
        static void clearAllSharedCacheTableLocks(Btree a) { }
        static void downgradeAllSharedCacheTableLocks(Btree a) { }
        static bool hasSharedCacheTableLock(Btree a, Pgno b, int c, int d) { return true; }
        static bool hasReadConflicts(Btree a, Pgno b) { return false; }
#endif

#if DEBUG
        internal static bool cursorHoldsMutex(BtCursor p) { return MutexEx.sqlite3_mutex_held(p.pBt.mutex); }
#else
        internal static bool cursorHoldsMutex(BtCursor p) { return true; }
#endif


#if !SQLITE_OMIT_INCRBLOB
static void invalidateOverflowCache(BtCursor pCur){
Debug.Assert( cursorHoldsMutex(pCur) );
//sqlite3_free(ref pCur.aOverflow);
pCur.aOverflow = null;
}

static void invalidateAllOverflowCache(BtShared pBt){
BtCursor p;
Debug.Assert( sqlite3_mutex_held(pBt.mutex) );
for(p=pBt.pCursor; p!=null; p=p.pNext){
invalidateOverflowCache(p);
}
}

static void invalidateIncrblobCursors(
Btree pBtree,          /* The database file to check */
i64 iRow,               /* The rowid that might be changing */
int isClearTable        /* True if all rows are being deleted */
){
BtCursor p;
BtShared pBt = pBtree.pBt;
Debug.Assert( sqlite3BtreeHoldsMutex(pBtree) );
for(p=pBt.pCursor; p!=null; p=p.pNext){
if( p.isIncrblobHandle && (isClearTable || p.info.nKey==iRow) ){
p.eState = CURSOR_INVALID;
}
}
}

#else
        internal static void invalidateOverflowCache(BtCursor pCur) { }
        internal static void invalidateAllOverflowCache(BtShared pBt) { }
        internal static void invalidateIncrblobCursors(Btree x, long y, int z) { }
#endif

        internal static SQLITE btreeSetHasContent(BtShared pBt, Pgno pgno)
        {
            var rc = SQLITE.OK;
            if (pBt.pHasContent == null)
            {
                Debug.Assert(pgno <= pBt.nPage);
                pBt.pHasContent = new Bitvec(pBt.nPage);
            }
            if (rc == SQLITE.OK && pgno <= pBt.pHasContent.sqlite3BitvecSize())
                rc = pBt.pHasContent.sqlite3BitvecSet(pgno);
            return rc;
        }

        internal static bool btreeGetHasContent(BtShared pBt, Pgno pgno)
        {
            var p = pBt.pHasContent;
            return (p != null && (pgno > p.sqlite3BitvecSize() || p.sqlite3BitvecTest(pgno) != 0));
        }

        internal static void btreeClearHasContent(BtShared pBt)
        {
            Bitvec.sqlite3BitvecDestroy(ref pBt.pHasContent);
            pBt.pHasContent = null;
        }

        internal static int saveCursorPosition(BtCursor pCur)
        {
            Debug.Assert(pCur.eState == CURSOR.VALID);
            Debug.Assert(pCur.pKey == null);
            Debug.Assert(cursorHoldsMutex(pCur));
            var rc = sqlite3BtreeKeySize(pCur, ref pCur.nKey);
            Debug.Assert(rc == SQLITE.OK);  // KeySize() cannot fail
            // If this is an intKey table, then the above call to BtreeKeySize() stores the integer key in pCur.nKey. In this case this value is
            // all that is required. Otherwise, if pCur is not open on an intKey table, then malloc space for and store the pCur.nKey bytes of key data.
            if (pCur.apPage[0].intKey == 0)
            {
                var pKey = MallocEx.sqlite3Malloc((int)pCur.nKey);
                rc = sqlite3BtreeKey(pCur, 0, (uint)pCur.nKey, pKey);
                if (rc == SQLITE.OK)
                    pCur.pKey = pKey;
            }
            Debug.Assert(pCur.apPage[0].intKey == 0 || pCur.pKey == null);
            if (rc == SQLITE.OK)
            {
                for (var i = 0; i <= pCur.iPage; i++)
                {
                    releasePage(pCur.apPage[i]);
                    pCur.apPage[i] = null;
                }
                pCur.iPage = -1;
                pCur.eState = CURSOR.REQUIRESEEK;
            }
            invalidateOverflowCache(pCur);
            return rc;
        }

        internal static SQLITE saveAllCursors(BtShared pBt, Pgno iRoot, BtCursor pExcept)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pExcept == null || pExcept.pBt == pBt);
            for (var p = pBt.pCursor; p != null; p = p.pNext)
                if (p != pExcept && (0 == iRoot || p.pgnoRoot == iRoot) && p.eState == CURSOR.VALID)
                {
                    var rc = saveCursorPosition(p);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            return SQLITE.OK;
        }

        internal static void sqlite3BtreeClearCursor(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            MallocEx.sqlite3_free(ref pCur.pKey);
            pCur.eState = CURSOR.INVALID;
        }

        internal static SQLITE btreeMoveto(BtCursor pCur, byte[] pKey, long nKey, int bias, ref int pRes)
        {
            UnpackedRecord pIdxKey;   // Unpacked index key
            var aSpace = new UnpackedRecord();
            if (pKey != null)
            {
                Debug.Assert(nKey == (long)(int)nKey);
                pIdxKey = sqlite3VdbeRecordUnpack(pCur.pKeyInfo, (int)nKey, pKey, aSpace, 16);
            }
            else
                pIdxKey = null;
            var rc = sqlite3BtreeMovetoUnpacked(pCur, pIdxKey, nKey, bias != 0 ? 1 : 0, ref pRes);
            if (pKey != null)
                sqlite3VdbeDeleteUnpackedRecord(pIdxKey);
            return rc;
        }

        internal static SQLITE btreeRestoreCursorPosition(BtCursor pCur)
        {
            Debug.Assert(cursorHoldsMutex(pCur));
            Debug.Assert(pCur.eState >= CURSOR.REQUIRESEEK);
            if (pCur.eState == CURSOR.FAULT)
                return pCur.skipNext;
            pCur.eState = CURSOR.INVALID;
            var rc = btreeMoveto(pCur, pCur.pKey, pCur.nKey, 0, ref pCur.skipNext);
            if (rc == SQLITE.OK)
            {
                pCur.pKey = null;
                Debug.Assert(pCur.eState == CURSOR.VALID || pCur.eState == CURSOR.INVALID);
            }
            return rc;
        }

        internal static SQLITE restoreCursorPosition(BtCursor pCur) { return (pCur.eState >= CURSOR.REQUIRESEEK ? btreeRestoreCursorPosition(pCur) : SQLITE.OK); }

        internal static SQLITE sqlite3BtreeCursorHasMoved(BtCursor pCur, ref int pHasMoved)
        {
            var rc = restoreCursorPosition(pCur);
            if (rc != SQLITE.OK)
            {
                pHasMoved = 1;
                return rc;
            }
            pHasMoved = (pCur.eState != CURSOR.VALID || pCur.skipNext != 0 ? 1 : 0);
            return SQLITE.OK;
        }

#if !SQLITE_OMIT_AUTOVACUUM
        internal static Pgno ptrmapPageno(BtShared pBt, Pgno pgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            if (pgno < 2)
                return 0;
            var nPagesPerMapPage = (int)(pBt.usableSize / 5 + 1);
            var iPtrMap = (Pgno)((pgno - 2) / nPagesPerMapPage);
            var ret = (Pgno)(iPtrMap * nPagesPerMapPage) + 2;
            if (ret == PENDING_BYTE_PAGE(pBt))
                ret++;
            return ret;
        }

        internal static void ptrmapPut(BtShared pBt, Pgno key, byte eType, Pgno parent, ref SQLITE pRC)
        {
            if (pRC != 0)
                return;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            // The master-journal page number must never be used as a pointer map page
            Debug.Assert(!PTRMAP_ISPAGE(pBt, PENDING_BYTE_PAGE(pBt)));
            Debug.Assert(pBt.autoVacuum);
            if (key == 0)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                return;
            }
            var iPtrmap = PTRMAP_PAGENO(pBt, key);
            var pDbPage = new PgHdr();  // The pointer map page 
            var rc = pBt.pPager.sqlite3PagerGet(iPtrmap, ref pDbPage);
            if (rc != SQLITE.OK)
            {
                pRC = rc;
                return;
            }
            var offset = (int)PTRMAP_PTROFFSET(iPtrmap, key);
            if (offset < 0)
            {
                pRC = SysEx.SQLITE_CORRUPT_BKPT();
                goto ptrmap_exit;
            }
            Debug.Assert(offset <= (int)pBt.usableSize - 5);
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage); // The pointer map data 
            if (eType != pPtrmap[offset] || ConvertEx.sqlite3Get4byte(pPtrmap, offset + 1) != parent)
            {
                TRACE("PTRMAP_UPDATE: {0}->({1},{2})", key, eType, parent);
                pRC = rc = Pager.sqlite3PagerWrite(pDbPage);
                if (rc == SQLITE.OK)
                {
                    pPtrmap[offset] = eType;
                    ConvertEx.sqlite3Put4byte(pPtrmap, offset + 1, parent);
                }
            }
        ptrmap_exit:
            Pager.sqlite3PagerUnref(pDbPage);
        }

        internal static SQLITE ptrmapGet(BtShared pBt, Pgno key, ref byte pEType, ref Pgno pPgno)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            var iPtrmap = (int)PTRMAP_PAGENO(pBt, key);
            var pDbPage = new PgHdr(); // The pointer map page
            var rc = pBt.pPager.sqlite3PagerGet((Pgno)iPtrmap, ref pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            var pPtrmap = Pager.sqlite3PagerGetData(pDbPage);// Pointer map page data
            var offset = (int)PTRMAP_PTROFFSET((Pgno)iPtrmap, key);
            if (offset < 0)
            {
                Pager.sqlite3PagerUnref(pDbPage);
                return SysEx.SQLITE_CORRUPT_BKPT();
            }
            Debug.Assert(offset <= (int)pBt.usableSize - 5);
            pEType = pPtrmap[offset];
            pPgno = ConvertEx.sqlite3Get4byte(pPtrmap, offset + 1);
            Pager.sqlite3PagerUnref(pDbPage);
            if (pEType < 1 || pEType > 5)
                return SysEx.SQLITE_CORRUPT_BKPT();
            return SQLITE.OK;
        }

#else
        internal static Pgno ptrmapPageno(BtShared pBt, Pgno pgno) { return 0; }
        internal static void ptrmapPut(BtShared pBt, Pgno key, byte eType, Pgno parent, ref SQLITE pRC) { pRC = SQLITE.OK; }
        internal static SQLITE ptrmapGet(BtShared pBt, Pgno key, ref byte pEType, ref Pgno pPgno) { return SQLITE.OK; }
#endif

        internal static void pageReinit(DbPage pData)
        {
            var pPage = Pager.sqlite3PagerGetExtra(pData);
            Debug.Assert(Pager.sqlite3PagerPageRefcount(pData) > 0);
            if (pPage.isInit != 0)
            {
                Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
                pPage.isInit = 0;
                if (Pager.sqlite3PagerPageRefcount(pData) > 1)
                    // pPage might not be a btree page;  it might be an overflow page or ptrmap page or a free page.  In those cases, the following
                    // call to btreeInitPage() will likely return SQLITE_CORRUPT. But no harm is done by this.  And it is very important that
                    // btreeInitPage() be called on every btree page so we make the call for every page that comes in for re-initing.
                    btreeInitPage(pPage);
            }
        }

        internal static int btreeInvokeBusyHandler(object pArg)
        {
            var pBt = (BtShared)pArg;
            Debug.Assert(pBt.db != null);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.db.mutex));
            return sqlite3InvokeBusyHandler(pBt.db.busyHandler);
        }

        internal static SQLITE sqlite3BtreeOpen(VirtualFileSystem pVfs, string zFilename, sqlite3 db, ref Btree ppBtree, OPEN flags, VFSOPEN vfsFlags)
        {
            BtShared pBt = null;          // Shared part of btree structure
            Btree p;                      // Handle to return
            sqlite3_mutex mutexOpen = null;  // Prevents a race condition.
            var rc = SQLITE.OK;
            byte nReserve;                   // Byte of unused space on each page
            var zDbHeader = new byte[100]; // Database header content
            // True if opening an ephemeral, temporary database */
            bool isTempDb = string.IsNullOrEmpty(zFilename);
            // Set the variable isMemdb to true for an in-memory database, or  false for a file-based database.
#if SQLITE_OMIT_MEMORYDB
            var isMemdb = false;
#else
            var isMemdb = (zFilename == ":memory:" || isTempDb && sqlite3TempInMemory(db));
#endif
            Debug.Assert(db != null);
            Debug.Assert(pVfs != null);
            Debug.Assert(MutexEx.sqlite3_mutex_held(db.mutex));
            Debug.Assert(((uint)flags & 0xff) == (uint)flags);   // flags fit in 8 bits
            // Only a BTREE_SINGLE database can be BTREE_UNORDERED
            Debug.Assert((flags & OPEN.UNORDERED) == 0 || (flags & OPEN.SINGLE) != 0);
            // A BTREE_SINGLE database is always a temporary and/or ephemeral
            Debug.Assert((flags & OPEN.SINGLE) == 0 || isTempDb);
            if ((db.flags & SQLITE_NoReadlock) != 0)
                flags |= OPEN.NO_READLOCK;
            if (isMemdb)
                flags |= OPEN.MEMORY;
            if ((vfsFlags & VFSOPEN.MAIN_DB) != 0 && (isMemdb || isTempDb))
                vfsFlags = (vfsFlags & ~VFSOPEN.MAIN_DB) | VFSOPEN.TEMP_DB;
            p = new Btree();
            p.inTrans = TRANS_NONE;
            p.db = db;
#if !SQLITE_OMIT_SHARED_CACHE
p.lock.pBtree = p;
p.lock.iTable = 1;
#endif
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
/*
** If this Btree is a candidate for shared cache, try to find an
** existing BtShared object that we can share with
*/
if( !isMemdb && !isTempDb ){
if( vfsFlags & SQLITE_OPEN_SHAREDCACHE ){
int nFullPathname = pVfs.mxPathname+1;
string zFullPathname = sqlite3Malloc(nFullPathname);
sqlite3_mutex *mutexShared;
p.sharable = 1;
if( !zFullPathname ){
p = null;//sqlite3_free(ref p);
return SQLITE_NOMEM;
}
sqlite3OsFullPathname(pVfs, zFilename, nFullPathname, zFullPathname);
mutexOpen = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_OPEN);
sqlite3_mutex_enter(mutexOpen);
mutexShared = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
sqlite3_mutex_enter(mutexShared);
for(pBt=GLOBAL(BtShared*,sqlite3SharedCacheList); pBt; pBt=pBt.pNext){
Debug.Assert( pBt.nRef>0 );
if( 0==strcmp(zFullPathname, sqlite3PagerFilename(pBt.pPager))
&& sqlite3PagerVfs(pBt.pPager)==pVfs ){
int iDb;
for(iDb=db.nDb-1; iDb>=0; iDb--){
Btree pExisting = db.aDb[iDb].pBt;
if( pExisting && pExisting.pBt==pBt ){
sqlite3_mutex_leave(mutexShared);
sqlite3_mutex_leave(mutexOpen);
zFullPathname = null;//sqlite3_free(ref zFullPathname);
p=null;//sqlite3_free(ref p);
return SQLITE_CONSTRAINT;
}
}
p.pBt = pBt;
pBt.nRef++;
break;
}
}
sqlite3_mutex_leave(mutexShared);
zFullPathname=null;//sqlite3_free(ref zFullPathname);
}
#if DEBUG
else{
/* In debug mode, we mark all persistent databases as sharable
** even when they are not.  This exercises the locking code and
** gives more opportunity for asserts(sqlite3_mutex_held())
** statements to find locking problems.
*/
p.sharable = 1;
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
                rc = Pager.sqlite3PagerOpen(pVfs, out pBt.pPager, zFilename, EXTRA_SIZE, flags, vfsFlags, pageReinit);
                if (rc == SQLITE.OK)
                    rc = pBt.pPager.sqlite3PagerReadFileheader(zDbHeader.Length, zDbHeader);
                if (rc != SQLITE.OK)
                    goto btree_open_out;
                pBt.openFlags = (byte)flags;
                pBt.db = db;
                pBt.pPager.sqlite3PagerSetBusyhandler(btreeInvokeBusyHandler, pBt);
                p.pBt = pBt;
                pBt.pCursor = null;
                pBt.pPage1 = null;
                pBt.readOnly = pBt.pPager.sqlite3PagerIsreadonly();
#if SQLITE_SECURE_DELETE
pBt.secureDelete = true;
#endif
                pBt.pageSize = (uint)((zDbHeader[16] << 8) | (zDbHeader[17] << 16)); if (pBt.pageSize < 512 || pBt.pageSize > Pager.SQLITE_MAX_PAGE_SIZE || ((pBt.pageSize - 1) & pBt.pageSize) != 0)
                {
                    pBt.pageSize = 0;
#if !SQLITE_OMIT_AUTOVACUUM
                    // If the magic name ":memory:" will create an in-memory database, then leave the autoVacuum mode at 0 (do not auto-vacuum), even if
                    // SQLITE_DEFAULT_AUTOVACUUM is true. On the other hand, if SQLITE_OMIT_MEMORYDB has been defined, then ":memory:" is just a
                    // regular file-name. In this case the auto-vacuum applies as per normal.
                    if (zFilename != string.Empty && !isMemdb)
                    {
                        pBt.autoVacuum = (SQLITE_DEFAULT_AUTOVACUUM != 0);
                        pBt.incrVacuum = (SQLITE_DEFAULT_AUTOVACUUM == 2);
                    }
#endif
                    nReserve = 0;
                }
                else
                {
                    nReserve = zDbHeader[20];
                    pBt.pageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
                    pBt.autoVacuum = ConvertEx.sqlite3Get4byte(zDbHeader, 36 + 4 * 4) != 0;
                    pBt.incrVacuum = ConvertEx.sqlite3Get4byte(zDbHeader, 36 + 7 * 4) != 0;
#endif
                }
                rc = pBt.pPager.sqlite3PagerSetPagesize(ref pBt.pageSize, nReserve);
                if (rc != SQLITE.OK)
                    goto btree_open_out;
                pBt.usableSize = (ushort)(pBt.pageSize - nReserve);
                Debug.Assert((pBt.pageSize & 7) == 0);  // 8-byte alignment of pageSize
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
/* Add the new BtShared object to the linked list sharable BtShareds.
*/
if( p.sharable ){
sqlite3_mutex *mutexShared;
pBt.nRef = 1;
mutexShared = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
if( SQLITE_THREADSAFE && sqlite3GlobalConfig.bCoreMutex ){
pBt.mutex = sqlite3MutexAlloc(SQLITE_MUTEX_FAST);
if( pBt.mutex==null ){
rc = SQLITE_NOMEM;
db.mallocFailed = 0;
goto btree_open_out;
}
}
sqlite3_mutex_enter(mutexShared);
pBt.pNext = GLOBAL(BtShared*,sqlite3SharedCacheList);
GLOBAL(BtShared*,sqlite3SharedCacheList) = pBt;
sqlite3_mutex_leave(mutexShared);
}
#endif
            }
#if !SQLITE_OMIT_SHARED_CACHE && !SQLITE_OMIT_DISKIO
/* If the new Btree uses a sharable pBtShared, then link the new
** Btree into the list of all sharable Btrees for the same connection.
** The list is kept in ascending order by pBt address.
*/
if( p.sharable ){
int i;
Btree pSib;
for(i=0; i<db.nDb; i++){
if( (pSib = db.aDb[i].pBt)!=null && pSib.sharable ){
while( pSib.pPrev ){ pSib = pSib.pPrev; }
if( p.pBt<pSib.pBt ){
p.pNext = pSib;
p.pPrev = 0;
pSib.pPrev = p;
}else{
while( pSib.pNext && pSib.pNext.pBt<p.pBt ){
pSib = pSib.pNext;
}
p.pNext = pSib.pNext;
p.pPrev = pSib;
if( p.pNext ){
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
            if (rc != SQLITE.OK)
            {
                if (pBt != null && pBt.pPager != null)
                    pBt.pPager.sqlite3PagerClose();
                pBt = null;
                p = null;
                ppBtree = null;
            }
            else
            {
                // If the B-Tree was successfully opened, set the pager-cache size to the default value. Except, when opening on an existing shared pager-cache,
                // do not change the pager-cache size.
                if (sqlite3BtreeSchema(p, 0, null) == null)
                    p.pBt.pPager.sqlite3PagerSetCachesize(SQLITE_DEFAULT_CACHE_SIZE);
            }
            if (mutexOpen != null)
            {
                Debug.Assert(MutexEx.sqlite3_mutex_held(mutexOpen));
                MutexEx.sqlite3_mutex_leave(mutexOpen);
            }
            return rc;
        }

        internal static bool removeFromSharingList(BtShared pBt)
        {
#if !SQLITE_OMIT_SHARED_CACHE
sqlite3_mutex pMaster;
BtShared pList;
bool removed = false;
Debug.Assert( sqlite3_mutex_notheld(pBt.mutex) );
pMaster = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
sqlite3_mutex_enter(pMaster);
pBt.nRef--;
if( pBt.nRef<=0 ){
if( GLOBAL(BtShared*,sqlite3SharedCacheList)==pBt ){
GLOBAL(BtShared*,sqlite3SharedCacheList) = pBt.pNext;
}else{
pList = GLOBAL(BtShared*,sqlite3SharedCacheList);
while( ALWAYS(pList) && pList.pNext!=pBt ){
pList=pList.pNext;
}
if( ALWAYS(pList) ){
pList.pNext = pBt.pNext;
}
}
if( SQLITE_THREADSAFE ){
sqlite3_mutex_free(pBt.mutex);
}
removed = true;
}
sqlite3_mutex_leave(pMaster);
return removed;
#else
            return true;
#endif
        }

        internal static void allocateTempSpace(BtShared pBt) { if (pBt.pTmpSpace == null) pBt.pTmpSpace = MallocEx.sqlite3Malloc((int)pBt.pageSize); }
        internal static void freeTempSpace(BtShared pBt) { Pager.sqlite3PageFree(ref pBt.pTmpSpace); }

        internal static SQLITE sqlite3BtreeClose(ref Btree p)
        {
            // Close all cursors opened via this handle.
            Debug.Assert(MutexEx.sqlite3_mutex_held(p.db.mutex));
            sqlite3BtreeEnter(p);
            var pBt = p.pBt;
            var pCur = pBt.pCursor;
            while (pCur != null)
            {
                var pTmp = pCur;
                pCur = pCur.pNext;
                if (pTmp.pBtree == p)
                    sqlite3BtreeCloseCursor(pTmp);
            }
            // Rollback any active transaction and free the handle structure. The call to sqlite3BtreeRollback() drops any table-locks held by this handle.
            sqlite3BtreeRollback(p);
            sqlite3BtreeLeave(p);
            // If there are still other outstanding references to the shared-btree structure, return now. The remainder of this procedure cleans up the shared-btree.
            Debug.Assert(p.wantToLock == 0 && !p.locked);
            if (!p.sharable || removeFromSharingList(pBt))
            {
                // The pBt is no longer on the sharing list, so we can access it without having to hold the mutex.
                // Clean out and delete the BtShared object.
                Debug.Assert(null == pBt.pCursor);
                pBt.pPager.sqlite3PagerClose();
                if (pBt.xFreeSchema != null && pBt.pSchema != null)
                    pBt.xFreeSchema(pBt.pSchema);
                pBt.pSchema = null;// sqlite3DbFree(0, pBt->pSchema);
                //freeTempSpace(pBt);
                pBt = null; //sqlite3_free(ref pBt);
            }
#if !SQLITE_OMIT_SHARED_CACHE
Debug.Assert( p.wantToLock==null );
Debug.Assert( p.locked==null );
if( p.pPrev ) p.pPrev.pNext = p.pNext;
if( p.pNext ) p.pNext.pPrev = p.pPrev;
#endif
            return SQLITE.OK;
        }

        internal static int sqlite3BtreeSetCacheSize(Btree p, int mxPage)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(p.db.mutex));
            sqlite3BtreeEnter(p);
            var pBt = p.pBt;
            sqlite3PagerSetCachesize(pBt.pPager, mxPage);
            sqlite3BtreeLeave(p);
            return SQLITE.OK;
        }

#if !SQLITE_OMIT_PAGER_PRAGMAS
        internal static SQLITE sqlite3BtreeSetSafetyLevel(Btree p, int level, int fullSync, int ckptFullSync)
        {
            var pBt = p.pBt;
            Debug.Assert(MutexEx.sqlite3_mutex_held(p.db.mutex));
            Debug.Assert(level >= 1 && level <= 3);
            sqlite3BtreeEnter(p);
            pBt.pPager.sqlite3PagerSetSafetyLevel(level, fullSync, ckptFullSync);
            sqlite3BtreeLeave(p);
            return SQLITE.OK;
        }
#endif

        internal static int sqlite3BtreeSyncDisabled(Btree p)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(p.db.mutex));
            sqlite3BtreeEnter(p);
            var pBt = p.pBt;
            Debug.Assert(pBt != null && pBt.pPager != null);
            var rc = (pBt.pPager.sqlite3PagerNosync() ? 1 : 0);
            sqlite3BtreeLeave(p);
            return rc;
        }

        internal static SQLITE sqlite3BtreeSetPageSize(Btree p, int pageSize, int nReserve, int iFix)
        {
            Debug.Assert(nReserve >= -1 && nReserve <= 255);
            sqlite3BtreeEnter(p);
            var pBt = p.pBt;
            if (pBt.pageSizeFixed)
            {
                sqlite3BtreeLeave(p);
                return SQLITE.READONLY;
            }
            if (nReserve < 0)
                nReserve = (int)(pBt.pageSize - pBt.usableSize);
            Debug.Assert(nReserve >= 0 && nReserve <= 255);
            if (pageSize >= 512 && pageSize <= Pager.SQLITE_MAX_PAGE_SIZE && ((pageSize - 1) & pageSize) == 0)
            {
                Debug.Assert((pageSize & 7) == 0);
                Debug.Assert(pBt.pPage1 == null && pBt.pCursor == null);
                pBt.pageSize = (uint)pageSize;
            }
            var rc = pBt.pPager.sqlite3PagerSetPagesize(ref pBt.pageSize, nReserve);
            pBt.usableSize = (ushort)(pBt.pageSize - nReserve);
            if (iFix != 0)
                pBt.pageSizeFixed = true;
            sqlite3BtreeLeave(p);
            return rc;
        }

        internal static int sqlite3BtreeGetPageSize(Btree p) { return (int)p.pBt.pageSize; }

#if !SQLITE_OMIT_PAGER_PRAGMAS || !SQLITE_OMIT_VACUUM
        internal static int sqlite3BtreeGetReserve(Btree p)
        {
            sqlite3BtreeEnter(p);
            var n = (int)(p.pBt.pageSize - p.pBt.usableSize);
            sqlite3BtreeLeave(p);
            return n;
        }

        internal static Pgno sqlite3BtreeMaxPageCount(Btree p, int mxPage)
        {
            sqlite3BtreeEnter(p);
            var n = (Pgno)p.pBt.pPager.sqlite3PagerMaxPageCount(mxPage);
            sqlite3BtreeLeave(p);
            return n;
        }

        internal static int sqlite3BtreeSecureDelete(Btree p, int newFlag)
        {
            if (p == null)
                return 0;
            sqlite3BtreeEnter(p);
            if (newFlag >= 0)
                p.pBt.secureDelete = (newFlag != 0);
            var b = (p.pBt.secureDelete ? 1 : 0);
            sqlite3BtreeLeave(p);
            return b;
        }
#endif

        internal static SQLITE sqlite3BtreeSetAutoVacuum(Btree p, int autoVacuum)
        {
#if SQLITE_OMIT_AUTOVACUUM
            return SQLITE.READONLY;
#else
            sqlite3BtreeEnter(p);
            var rc = SQLITE.OK;
            var av = (byte)autoVacuum;
            var pBt = p.pBt;
            if (pBt.pageSizeFixed && (av != 0) != pBt.autoVacuum)
                rc = SQLITE.READONLY;
            else
            {
                pBt.autoVacuum = av != 0;
                pBt.incrVacuum = av == 2;
            }
            sqlite3BtreeLeave(p);
            return rc;
#endif
        }

        internal static int sqlite3BtreeGetAutoVacuum(Btree p)
        {
#if SQLITE_OMIT_AUTOVACUUM
return BTREE.AUTOVACUUM_NONE;
#else
            sqlite3BtreeEnter(p);
            var rc = ((!p.pBt.autoVacuum) ? BTREE_AUTOVACUUM_NONE : (!p.pBt.incrVacuum) ? BTREE_AUTOVACUUM_FULL : BTREE_AUTOVACUUM_INCR);
            sqlite3BtreeLeave(p);
            return rc;
#endif
        }


        internal static SQLITE lockBtree(BtShared pBt)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pBt.pPage1 == null);
            var rc = pBt.pPager.sqlite3PagerSharedLock();
            if (rc != SQLITE.OK)
                return rc;
            MemPage pPage1 = null; // Page 1 of the database file
            rc = btreeGetPage(pBt, 1, ref pPage1, 0);
            if (rc != SQLITE.OK)
                return rc;
            // Do some checking to help insure the file we opened really is a valid database file.
            Pgno nPageHeader;      // Number of pages in the database according to hdr
            var nPage = nPageHeader = ConvertEx.sqlite3Get4byte(pPage1.aData, 28); // Number of pages in the database
            Pgno nPageFile;    // Number of pages in the database file
            pBt.pPager.sqlite3PagerPagecount(out nPageFile);
            if (nPage == 0 || ArrayEx.Compare(pPage1.aData, 24, pPage1.aData, 92, 4) != 0)
                nPage = nPageFile;
            if (nPage > 0)
            {
                var page1 = pPage1.aData;
                rc = SQLITE.NOTADB;
                if (ArrayEx.Compare(page1, zMagicHeader, 16) != 0)
                    goto page1_init_failed;
#if SQLITE_OMIT_WAL
                if (page1[18] > 1)
                    pBt.readOnly = true;
                if (page1[19] > 1)
                {
                    pBt.pSchema.file_format = page1[19];
                    goto page1_init_failed;
                }
#else
if( page1[18]>2 ){
pBt.readOnly = true;
}
if( page1[19]>2 ){
goto page1_init_failed;
}

/* If the write version is set to 2, this database should be accessed
** in WAL mode. If the log is not already open, open it now. Then 
** return SQLITE_OK and return without populating BtShared.pPage1.
** The caller detects this and calls this function again. This is
** required as the version of page 1 currently in the page1 buffer
** may not be the latest version - there may be a newer one in the log
** file.
*/
if( page1[19]==2 && pBt.doNotUseWAL==false ){
int isOpen = 0;
rc = sqlite3PagerOpenWal(pBt.pPager, ref isOpen);
if( rc!=SQLITE_OK ){
goto page1_init_failed;
}else if( isOpen==0 ){
releasePage(pPage1);
return SQLITE_OK;
}
rc = SQLITE_NOTADB;
}
#endif

                // The maximum embedded fraction must be exactly 25%.  And the minimum embedded fraction must be 12.5% for both leaf-data and non-leaf-data.
                // The original design allowed these amounts to vary, but as of version 3.6.0, we require them to be fixed.
                if (ArrayEx.Compare(page1, 21, "\x0040\x0020\x0020", 3) != 0)//   "\100\040\040"
                    goto page1_init_failed;
                var pageSize = (uint)((page1[16] << 8) | (page1[17] << 16));
                if (((pageSize - 1) & pageSize) != 0 || pageSize > Pager.SQLITE_MAX_PAGE_SIZE || pageSize <= 256)
                    goto page1_init_failed;
                Debug.Assert((pageSize & 7) == 0);
                var usableSize = pageSize - page1[20];
                if (pageSize != pBt.pageSize)
                {
                    // After reading the first page of the database assuming a page size of BtShared.pageSize, we have discovered that the page-size is
                    // actually pageSize. Unlock the database, leave pBt.pPage1 at zero and return SQLITE_OK. The caller will call this function
                    // again with the correct page-size.
                    releasePage(pPage1);
                    pBt.usableSize = usableSize;
                    pBt.pageSize = pageSize;
                    rc = pBt.pPager.sqlite3PagerSetPagesize(ref pBt.pageSize, (int)(pageSize - usableSize));
                    return rc;
                }
                if ((pBt.db.flags & SQLITE_RecoveryMode) == 0 && nPage > nPageFile)
                {
                    rc = SysEx.SQLITE_CORRUPT_BKPT();
                    goto page1_init_failed;
                }
                if (usableSize < 480)
                    goto page1_init_failed;
                pBt.pageSize = pageSize;
                pBt.usableSize = usableSize;
#if !SQLITE_OMIT_AUTOVACUUM
                pBt.autoVacuum = (ConvertEx.sqlite3Get4byte(page1, 36 + 4 * 4) != 0);
                pBt.incrVacuum = (ConvertEx.sqlite3Get4byte(page1, 36 + 7 * 4) != 0);
#endif
            }
            // maxLocal is the maximum amount of payload to store locally for a cell.  Make sure it is small enough so that at least minFanout
            // cells can will fit on one page.  We assume a 10-byte page header. Besides the payload, the cell must store:
            //     2-byte pointer to the cell
            //     4-byte child pointer
            //     9-byte nKey value
            //     4-byte nData value
            //     4-byte overflow page pointer
            // So a cell consists of a 2-byte pointer, a header which is as much as 17 bytes long, 0 to N bytes of payload, and an optional 4 byte overflow page pointer.
            pBt.maxLocal = (ushort)((pBt.usableSize - 12) * 64 / 255 - 23);
            pBt.minLocal = (ushort)((pBt.usableSize - 12) * 32 / 255 - 23);
            pBt.maxLeaf = (ushort)(pBt.usableSize - 35);
            pBt.minLeaf = (ushort)((pBt.usableSize - 12) * 32 / 255 - 23);
            Debug.Assert(pBt.maxLeaf + 23 <= MX_CELL_SIZE(pBt));
            pBt.pPage1 = pPage1;
            pBt.nPage = nPage;
            return SQLITE.OK;
        page1_init_failed:
            releasePage(pPage1);
            pBt.pPage1 = null;
            return rc;
        }

        internal static void unlockBtreeIfUnused(BtShared pBt)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pBt.pCursor == null || pBt.inTransaction > TRANS_NONE);
            if (pBt.inTransaction == TRANS_NONE && pBt.pPage1 != null)
            {
                Debug.Assert(pBt.pPage1.aData != null);
                //Debug.Assert(pBt.pPager.sqlite3PagerRefcount() == 1 );
                releasePage(pBt.pPage1);
                pBt.pPage1 = null;
            }
        }

        internal static SQLITE newDatabase(BtShared pBt)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            if (pBt.nPage > 0)
                return SQLITE.OK;
            var pP1 = pBt.pPage1;
            Debug.Assert(pP1 != null);
            var data = pP1.aData;
            var rc = Pager.sqlite3PagerWrite(pP1.pDbPage);
            if (rc != SQLITE.OK)
                return rc;
            Buffer.BlockCopy(zMagicHeader, 0, data, 0, 16);
            Debug.Assert(zMagicHeader.Length == 16);
            data[16] = (byte)((pBt.pageSize >> 8) & 0xff);
            data[17] = (byte)((pBt.pageSize >> 16) & 0xff);
            data[18] = 1;
            data[19] = 1;
            Debug.Assert(pBt.usableSize <= pBt.pageSize && pBt.usableSize + 255 >= pBt.pageSize);
            data[20] = (byte)(pBt.pageSize - pBt.usableSize);
            data[21] = 64;
            data[22] = 32;
            data[23] = 32;
            zeroPage(pP1, PTF_INTKEY | PTF_LEAF | PTF_LEAFDATA);
            pBt.pageSizeFixed = true;
#if !SQLITE_OMIT_AUTOVACUUM
            Debug.Assert(pBt.autoVacuum == true || pBt.autoVacuum == false);
            Debug.Assert(pBt.incrVacuum == true || pBt.incrVacuum == false);
            ConvertEx.sqlite3Put4byte(data, 36 + 4 * 4, pBt.autoVacuum ? 1 : 0);
            ConvertEx.sqlite3Put4byte(data, 36 + 7 * 4, pBt.incrVacuum ? 1 : 0);
#endif
            pBt.nPage = 1;
            data[31] = 1;
            return SQLITE.OK;
        }

        internal static int sqlite3BtreeBeginTrans(Btree p, int wrflag)
        {
            var pBt = p.pBt;
            var rc = SQLITE.OK;
            sqlite3BtreeEnter(p);
            btreeIntegrity(p);
            // If the btree is already in a write-transaction, or it is already in a read-transaction and a read-transaction is requested, this is a no-op.
            if (p.inTrans == TRANS_WRITE || (p.inTrans == TRANS_READ && 0 == wrflag))
                goto trans_begun;
            // Write transactions are not possible on a read-only database
            if (pBt.readOnly && wrflag != 0)
            {
                rc = SQLITE.READONLY;
                goto trans_begun;
            }
#if !SQLITE_OMIT_SHARED_CACHE
/* If another database handle has already opened a write transaction
** on this shared-btree structure and a second write transaction is
** requested, return SQLITE_LOCKED.
*/
if( (wrflag && pBt.inTransaction==TRANS_WRITE) || pBt.isPending ){
sqlite3 pBlock = pBt.pWriter.db;
}else if( wrflag>1 ){
BtLock pIter;
for(pIter=pBt.pLock; pIter; pIter=pIter.pNext){
if( pIter.pBtree!=p ){
pBlock = pIter.pBtree.db;
break;
}
}
}
if( pBlock ){
sqlite3ConnectionBlocked(p.db, pBlock);
rc = SQLITE_LOCKED_SHAREDCACHE;
goto trans_begun;
}
#endif

            // Any read-only or read-write transaction implies a read-lock on page 1. So if some other shared-cache client already has a write-lock
            // on page 1, the transaction cannot be opened. */
            rc = querySharedCacheTableLock(p, MASTER_ROOT, READ_LOCK);
            if (rc != SQLITE.OK)
                goto trans_begun;
            pBt.initiallyEmpty = pBt.nPage == 0;
            do
            {
                // Call lockBtree() until either pBt.pPage1 is populated or lockBtree() returns something other than SQLITE_OK. lockBtree()
                // may return SQLITE_OK but leave pBt.pPage1 set to 0 if after reading page 1 it discovers that the page-size of the database
                // file is not pBt.pageSize. In this case lockBtree() will update pBt.pageSize to the page-size of the file on disk.
                while (pBt.pPage1 == null && (rc = lockBtree(pBt)) == SQLITE.OK) ;
                if (rc == SQLITE.OK && wrflag != 0)
                {
                    if (pBt.readOnly)
                        rc = SQLITE.READONLY;
                    else
                    {
                        rc = pBt.pPager.sqlite3PagerBegin(wrflag > 1, sqlite3TempInMemory(p.db) ? 1 : 0);
                        if (rc == SQLITE.OK)
                            rc = newDatabase(pBt);
                    }
                }
                if (rc != SQLITE.OK)
                    unlockBtreeIfUnused(pBt);
            } while (((int)rc & 0xFF) == (int)SQLITE.BUSY && pBt.inTransaction == TRANS_NONE && btreeInvokeBusyHandler(pBt) != 0);
            if (rc == SQLITE.OK)
            {
                if (p.inTrans == TRANS_NONE)
                {
                    pBt.nTransaction++;
#if !SQLITE_OMIT_SHARED_CACHE
if( p.sharable ){
Debug.Assert( p.lock.pBtree==p && p.lock.iTable==1 );
p.lock.eLock = READ_LOCK;
p.lock.pNext = pBt.pLock;
pBt.pLock = &p.lock;
}
#endif
                }
                p.inTrans = (wrflag != 0 ? TRANS_WRITE : TRANS_READ);
                if (p.inTrans > pBt.inTransaction)
                    pBt.inTransaction = p.inTrans;
                if (wrflag != 0)
                {
                    var pPage1 = pBt.pPage1;
#if !SQLITE_OMIT_SHARED_CACHE
Debug.Assert( !pBt.pWriter );
pBt.pWriter = p;
pBt.isExclusive = (u8)(wrflag>1);
#endif
                    // If the db-size header field is incorrect (as it may be if an old client has been writing the database file), update it now. Doing
                    // this sooner rather than later means the database size can safely  re-read the database size from page 1 if a savepoint or transaction
                    // rollback occurs within the transaction.
                    if (pBt.nPage != ConvertEx.sqlite3Get4byte(pPage1.aData, 28))
                    {
                        rc = Pager.sqlite3PagerWrite(pPage1.pDbPage);
                        if (rc == SQLITE.OK)
                            Convert.sqlite3Put4byte(pPage1.aData, (u32)28, pBt.nPage);
                    }
                }
            }
        trans_begun:
            if (rc == SQLITE.OK && wrflag != 0)
                // This call makes sure that the pager has the correct number of open savepoints. If the second parameter is greater than 0 and
                // the sub-journal is not already open, then it will be opened here.
                rc = sqlite3PagerOpenSavepoint(pBt.pPager, p.db.nSavepoint);
            btreeIntegrity(p);
            sqlite3BtreeLeave(p);
            return rc;
        }

#if !SQLITE_OMIT_AUTOVACUUM
        internal static int setChildPtrmaps(MemPage pPage)
        {
            var pBt = pPage.pBt;
            var isInitOrig = pPage.isInit;
            var pgno = pPage.pgno;
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            var rc = btreeInitPage(pPage);
            if (rc != SQLITE.OK)
                goto set_child_ptrmaps_out;
            var nCell = pPage.nCell; // Number of cells in page pPage
            for (var i = 0; i < nCell; i++)
            {
                int pCell = findCell(pPage, i);
                ptrmapPutOvflPtr(pPage, pCell, ref rc);
                if (pPage.leaf == 0)
                {
                    Pgno childPgno = sqlite3Get4byte(pPage.aData, pCell);
                    ptrmapPut(pBt, childPgno, PTRMAP_BTREE, pgno, ref rc);
                }
            }
            if (pPage.leaf == 0)
            {
                Pgno childPgno = sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8);
                ptrmapPut(pBt, childPgno, PTRMAP_BTREE, pgno, ref rc);
            }
        set_child_ptrmaps_out:
            pPage.isInit = isInitOrig;
            return rc;
        }

        internal static int modifyPagePointer(MemPage pPage, Pgno iFrom, Pgno iTo, byte eType)
        {
            Debug.Assert(MutexEx.sqlite3_mutex_held(pPage.pBt.mutex));
            Debug.Assert(Pager.sqlite3PagerIswriteable(pPage.pDbPage));
            if (eType == PTRMAP_OVERFLOW2)
            {
                // The pointer is always the first 4 bytes of the page in this case.
                if (ConvertEx.sqlite3Get4byte(pPage.aData) != iFrom)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                ConvertEx.sqlite3Put4byte(pPage.aData, iTo);
            }
            else
            {
                var isInitOrig = pPage.isInit;
                btreeInitPage(pPage);
                var nCell = pPage.nCell;
                for (var i = 0; i < nCell; i++)
                {
                    var pCell = findCell(pPage, i);
                    if (eType == PTRMAP_OVERFLOW1)
                    {
                        var info = new CellInfo();
                        btreeParseCellPtr(pPage, pCell, ref info);
                        if (info.iOverflow != 0)
                            if (iFrom == ConvertEx.sqlite3Get4byte(pPage.aData, pCell, info.iOverflow))
                            {
                                ConvertEx.sqlite3Put4byte(pPage.aData, pCell + info.iOverflow, (int)iTo);
                                break;
                            }
                    }
                    else
                    {
                        if (ConvertEx.sqlite3Get4byte(pPage.aData, pCell) == iFrom)
                        {
                            ConvertEx.sqlite3Put4byte(pPage.aData, pCell, (int)iTo);
                            break;
                        }
                    }
                }
                if (i == nCell)
                {
                    if (eType != PTRMAP_BTREE || ConvertEx.sqlite3Get4byte(pPage.aData, pPage.hdrOffset + 8) != iFrom)
                        return SysEx.SQLITE_CORRUPT_BKPT();
                    ConvertEx.sqlite3Put4byte(pPage.aData, pPage.hdrOffset + 8, iTo);
                }
                pPage.isInit = isInitOrig;
            }
            return SQLITE.OK;
        }

        internal static SQLITE relocatePage(BtShared pBt, MemPage pDbPage, byte eType, Pgno iPtrPage, Pgno iFreePage, int isCommit)
        {
            var pPtrPage = new MemPage();   // The page that contains a pointer to pDbPage
            var iDbPage = pDbPage.pgno;
            var pPager = pBt.pPager;
            Debug.Assert(eType == PTRMAP_OVERFLOW2 || eType == PTRMAP_OVERFLOW1 || eType == PTRMAP_BTREE || eType == PTRMAP_ROOTPAGE);
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(pDbPage.pBt == pBt);
            // Move page iDbPage from its current location to page number iFreePage
            TRACE("AUTOVACUUM: Moving %d to free page %d (ptr page %d type %d)\n", iDbPage, iFreePage, iPtrPage, eType);
            var rc = sqlite3PagerMovepage(pPager, pDbPage.pDbPage, iFreePage, isCommit);
            if (rc != SQLITE.OK)
                return rc;
            pDbPage.pgno = iFreePage;
            // If pDbPage was a btree-page, then it may have child pages and/or cells that point to overflow pages. The pointer map entries for all these
            // pages need to be changed.
            // If pDbPage is an overflow page, then the first 4 bytes may store a pointer to a subsequent overflow page. If this is the case, then
            // the pointer map needs to be updated for the subsequent overflow page.
            if (eType == PTRMAP_BTREE || eType == PTRMAP_ROOTPAGE)
            {
                rc = setChildPtrmaps(pDbPage);
                if (rc != SQLITE.OK)
                    return rc;
            }
            else
            {
                var nextOvfl = (Pgno)ConvertEx.sqlite3Get4byte(pDbPage.aData);
                if (nextOvfl != 0)
                {
                    ptrmapPut(pBt, nextOvfl, PTRMAP_OVERFLOW2, iFreePage, ref rc);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            }
            // Fix the database pointer on page iPtrPage that pointed at iDbPage so that it points at iFreePage. Also fix the pointer map entry for iPtrPage.
            if (eType != PTRMAP_ROOTPAGE)
            {
                rc = btreeGetPage(pBt, iPtrPage, ref pPtrPage, 0);
                if (rc != SQLITE.OK)
                    return rc;
                rc = Pager.sqlite3PagerWrite(pPtrPage.pDbPage);
                if (rc != SQLITE.OK)
                {
                    releasePage(pPtrPage);
                    return rc;
                }
                rc = modifyPagePointer(pPtrPage, iDbPage, iFreePage, eType);
                releasePage(pPtrPage);
                if (rc == SQLITE.OK)
                    ptrmapPut(pBt, iFreePage, eType, iPtrPage, ref rc);
            }
            return rc;
        }

        internal static int incrVacuumStep(BtShared pBt, Pgno nFin, Pgno iLastPg)
        {
            Pgno nFreeList;           // Number of pages still on the free-list
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            Debug.Assert(iLastPg > nFin);
            if (!PTRMAP_ISPAGE(pBt, iLastPg) && iLastPg != PENDING_BYTE_PAGE(pBt))
            {
                byte eType = 0;
                Pgno iPtrPage = 0;
                nFreeList = ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 36);
                if (nFreeList == 0)
                    return SQLITE.DONE;
                rc = ptrmapGet(pBt, iLastPg, ref eType, ref iPtrPage);
                if (rc != SQLITE.OK)
                    return rc;
                if (eType == PTRMAP_ROOTPAGE)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                if (eType == PTRMAP_FREEPAGE)
                {
                    if (nFin == 0)
                    {
                        // Remove the page from the files free-list. This is not required if nFin is non-zero. In that case, the free-list will be
                        // truncated to zero after this function returns, so it doesn't matter if it still contains some garbage entries.
                        Pgno iFreePg = 0;
                        var pFreePg = new MemPage();
                        rc = allocateBtreePage(pBt, ref pFreePg, ref iFreePg, iLastPg, 1);
                        if (rc != SQLITE.OK)
                            return rc;
                        Debug.Assert(iFreePg == iLastPg);
                        releasePage(pFreePg);
                    }
                }
                else
                {
                    Pgno iFreePg = 0; // Index of free page to move pLastPg to
                    var pLastPg = new MemPage();
                    rc = btreeGetPage(pBt, iLastPg, ref pLastPg, 0);
                    if (rc != SQLITE.OK)
                        return rc;
                    // If nFin is zero, this loop runs exactly once and page pLastPg is swapped with the first free page pulled off the free list.
                    // On the other hand, if nFin is greater than zero, then keep looping until a free-page located within the first nFin pages of the file is found.
                    do
                    {
                        var pFreePg = new MemPage();
                        rc = allocateBtreePage(pBt, ref pFreePg, ref iFreePg, 0, 0);
                        if (rc != SQLITE.OK)
                        {
                            releasePage(pLastPg);
                            return rc;
                        }
                        releasePage(pFreePg);
                    } while (nFin != 0 && iFreePg > nFin);
                    Debug.Assert(iFreePg < iLastPg);
                    rc = Pager.sqlite3PagerWrite(pLastPg.pDbPage);
                    if (rc == SQLITE.OK)
                        rc = relocatePage(pBt, pLastPg, eType, iPtrPage, iFreePg, (nFin != 0) ? 1 : 0);
                    releasePage(pLastPg);
                    if (rc != SQLITE.OK)
                        return rc;
                }
            }
            if (nFin == 0)
            {
                iLastPg--;
                while (iLastPg == PENDING_BYTE_PAGE(pBt) || PTRMAP_ISPAGE(pBt, iLastPg))
                {
                    if (PTRMAP_ISPAGE(pBt, iLastPg))
                    {
                        var pPg = new MemPage();
                        rc = btreeGetPage(pBt, iLastPg, ref pPg, 0);
                        if (rc != SQLITE.OK)
                            return rc;
                        rc = sqlite3PagerWrite(pPg.pDbPage);
                        releasePage(pPg);
                        if (rc != SQLITE.OK)
                            return rc;
                    }
                    iLastPg--;
                }
                sqlite3PagerTruncateImage(pBt.pPager, iLastPg);
                pBt.nPage = iLastPg;
            }
            return SQLITE.OK;
        }

        internal static int sqlite3BtreeIncrVacuum(Btree p)
        {
            var pBt = p.pBt;
            sqlite3BtreeEnter(p);
            Debug.Assert(pBt.inTransaction == TRANS_WRITE && p.inTrans == TRANS_WRITE);
            SQLITE rc;
            if (!pBt.autoVacuum)
                rc = SQLITE.DONE;
            else
            {
                invalidateAllOverflowCache(pBt);
                rc = incrVacuumStep(pBt, 0, btreePagecount(pBt));
                if (rc == SQLITE.OK)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.pPage1.pDbPage);
                    ConvertEx.sqlite3Put4byte(pBt.pPage1.aData, (u32)28, pBt.nPage);
                }
            }
            sqlite3BtreeLeave(p);
            return rc;
        }

        internal static int autoVacuumCommit(BtShared pBt)
        {
            var rc = SQLITE.OK;
            var pPager = pBt.pPager;
#if DEBUG
            int nRef = pPager.sqlite3PagerRefcount();
#else
            int nRef = 0;
#endif
            Debug.Assert(MutexEx.sqlite3_mutex_held(pBt.mutex));
            invalidateAllOverflowCache(pBt);
            Debug.Assert(pBt.autoVacuum);
            if (!pBt.incrVacuum)
            {
                var nOrig = btreePagecount(pBt); // Database size before freeing
                if (PTRMAP_ISPAGE(pBt, nOrig) || nOrig == PENDING_BYTE_PAGE(pBt))
                    // It is not possible to create a database for which the final page is either a pointer-map page or the pending-byte page. If one
                    // is encountered, this indicates corruption.
                    return SysEx.SQLITE_CORRUPT_BKPT();
                var nFree = (Pgno)ConvertEx.sqlite3Get4byte(pBt.pPage1.aData, 36); // Number of pages on the freelist initially
                var nEntry = (int)pBt.usableSize / 5; // Number of entries on one ptrmap page
                nPtrmap = (Pgno)((nFree - nOrig + PTRMAP_PAGENO(pBt, nOrig) + (Pgno)nEntry) / nEntry); // Number of PtrMap pages to be freed 
                nFin = nOrig - nFree - nPtrmap; // Number of pages in database after autovacuuming
                if (nOrig > PENDING_BYTE_PAGE(pBt) && nFin < PENDING_BYTE_PAGE(pBt))
                    nFin--;
                while (PTRMAP_ISPAGE(pBt, nFin) || nFin == PENDING_BYTE_PAGE(pBt))
                    nFin--;
                if (nFin > nOrig)
                    return SysEx.SQLITE_CORRUPT_BKPT();
                for (var iFree = nOrig; iFree > nFin && rc == SQLITE.OK; iFree--)
                    rc = incrVacuumStep(pBt, nFin, iFree);
                if ((rc == SQLITE.DONE || rc == SQLITE.OK) && nFree > 0)
                {
                    rc = Pager.sqlite3PagerWrite(pBt.pPage1.pDbPage);
                    Pager.sqlite3Put4byte(pBt.pPage1.aData, 32, 0);
                    Pager.sqlite3Put4byte(pBt.pPage1.aData, 36, 0);
                    Pager.sqlite3Put4byte(pBt.pPage1.aData, (u32)28, nFin);
                    pBt.pPager.sqlite3PagerTruncateImage(nFin);
                    pBt.nPage = nFin;
                }
                if (rc != SQLITE.OK)
                    sqlite3PagerRollback(pPager);
            }
            Debug.Assert(nRef == pPager.sqlite3PagerRefcount());
            return rc;
        }
#else
        internal static int setChildPtrmaps(MemPage pPage) { return SQLITE.OK; }
#endif
    }
}
