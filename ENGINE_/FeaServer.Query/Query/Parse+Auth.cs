using System;
using FeaServer.Auth;
using FeaServer.Core;
namespace FeaServer.Query
{
    public partial class Parse
    {
        private struct AuthContext
        {
            public Parse Parse;
            public string Context;
        }

        private int SetAuthorizer(Context ctx, Authorizer authorizer, object arg)
        {
            ctx.Mutex.Enter();
            ctx.Authorizer = authorizer;
            ctx.AuthorizerArg = arg;
            //ctx.ExpirePreparedStatements();
            ctx.Mutex.Leave();
            return (int)RC.OK;
        }

        private void AuthBadReturnCode()
        {
            ErrorMsg("authorizer malfunction");
            _rc = (int)RC.ERROR;
        }

        private int AuthReadCol(string table, string column, int db)
        {
            var database = _db.aDb[db].Name;
            var rc = _db.Authorizer(_db.AuthorizerArg, AuthorizerAC.READ, table, column, database, _authContext);
            if (rc == (int)AuthorizerRC.DENY)
            {
                if (_db.nDb > 2 || db != 0)
                    ErrorMsg("access to {0}.{1}.{2} is prohibited", database, table, column);
                else
                    ErrorMsg("access to {0}.{1} is prohibited", table, column);
                _rc = (int)RC.AUTH;
            }
            else if (rc != (int)AuthorizerRC.IGNORE && rc != (int)RC.OK)
                AuthBadReturnCode();
            return rc;
        }

        private void AuthRead(Expr Expr, Schema Schema, SrcList TabList)
        {
            //sqlite3 *db = pParse->db;
            //Table *pTab = 0;      /* The table being read */
            //const char *zCol;     /* Name of the column of the table */
            //int iSrc;             /* Index in pTabList->a[] of table being read */
            //int iDb;              /* The index of the database the expression refers to */
            //int iCol;             /* Index of column in table */

            //if( db->xAuth==0 ) return;
            //iDb = sqlite3SchemaToIndex(pParse->db, pSchema);
            //if( iDb<0 ){
            //  /* An attempt to read a column out of a subquery or other
            //  ** temporary table. */
            //  return;
            //}

            //assert( pExpr->op==TK_COLUMN || pExpr->op==TK_TRIGGER );
            //if( pExpr->op==TK_TRIGGER ){
            //  pTab = pParse->pTriggerTab;
            //}else{
            //  assert( pTabList );
            //  for(iSrc=0; ALWAYS(iSrc<pTabList->nSrc); iSrc++){
            //    if( pExpr->iTable==pTabList->a[iSrc].iCursor ){
            //      pTab = pTabList->a[iSrc].pTab;
            //      break;
            //    }
            //  }
            //}
            //iCol = pExpr->iColumn;
            //if( NEVER(pTab==0) ) return;

            //if( iCol>=0 ){
            //  assert( iCol<pTab->nCol );
            //  zCol = pTab->aCol[iCol].zName;
            //}else if( pTab->iPKey>=0 ){
            //  assert( pTab->iPKey<pTab->nCol );
            //  zCol = pTab->aCol[pTab->iPKey].zName;
            //}else{
            //  zCol = "ROWID";
            //}
            //assert( iDb>=0 && iDb<db->nDb );
            //if( SQLITE_IGNORE==sqlite3AuthReadCol(pParse, pTab->zName, zCol, iDb) ){
            //  pExpr->op = TK_NULL;
            //}
        }

        private int AuthCheck(AuthorizerAC code, string arg1, string arg2, string arg3)
        {
            //// Don't do any authorization checks if the database is initialising or if the parser is being invoked from within sqlite3_declare_vtab.
            //if (db->init.busy || IN_DECLARE_VTAB)
            //    return (int)RC.OK;
            //
            if (_db.Authorizer == null)
                return (int)RC.OK;
            var rc = _db.Authorizer(_db.AuthorizerArg, code, arg1, arg2, arg3, _authContext);
            if (rc == (int)AuthorizerRC.DENY)
            {
                ErrorMsg("not authorized");
                _rc = (int)RC.AUTH;
            }
            else if (rc != (int)RC.OK && rc != (int)AuthorizerRC.IGNORE)
            {
                rc = (int)AuthorizerRC.DENY;
                AuthBadReturnCode();
            }
            return rc;
        }

        private void AuthContextPush(AuthContext ctx, string authContext)
        {
            ctx.Parse = this;
            ctx.Context = _authContext;
            _authContext = authContext;
        }

        private void AuthContextPop(AuthContext ctx)
        {
            if (ctx.Parse != null)
            {
                ctx.Parse._authContext = ctx.Context;
                ctx.Parse = null;
            }
        }
    }
}
