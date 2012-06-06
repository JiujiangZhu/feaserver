namespace Contoso.Core
{
    public partial class Btree
    {
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
    }
}
