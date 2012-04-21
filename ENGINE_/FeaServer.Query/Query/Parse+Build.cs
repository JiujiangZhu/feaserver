using System.Diagnostics;
using System;
using TK = FeaServer.Query.Parser.TK;
using FeaServer.Auth;
namespace FeaServer.Query
{
    public partial class Parse
    {
        #region Parse

        public void BeginParse(byte explainFlag)
        {
            _explain = explainFlag;
            _nVar = 0;
        }

        public void NestedParse(string format, params object[] args)
        {
            if (_nErr > 0) return;
            Debug.Assert(_nested < 10);  // Nesting should only be of limited depth
            var sql = string.Format(format, args);
            _nested++;
            var savedState = GetAndResetState();
            string errorMessage;
            Tokenize.RunParser(this, sql, out errorMessage);
            RestoreState(savedState);
            _nested--;
        }
        public void FinishCoding() { }

        #endregion

        #region Transaction

        public void BeginTransaction(int type)
        {
            Debug.Assert(_db != null);
            if (AuthCheck(AuthorizerAC.TRANSACTION, "BEGIN", null, null) > 0)
                return;
            var v = GetVdbe();
            if (v == null) return;
            if (type != (int)TK.DEFERRED)
                for (var i = 0; i < _db.nDb; i++)
                {
                    v.AddOp2(OP_Transaction, i, (type == TK.EXCLUSIVE ? 1 : 0) + 1);
                    //v.UsesBtree(i);
                }
            v.AddOp2(OP_AutoCommit, 0, 0);
        }

        public void CommitTransaction()
        {
            Debug.Assert(_db != null);
            if (AuthCheck(AuthorizerAC.TRANSACTION, "COMMIT", null, null) > 0)
                return;
            var v = GetVdbe();
            if (v != null)
                v.AddOp2(OP_AutoCommit, 1, 0);
        }

        public void RollbackTransaction()
        {
            Debug.Assert(_db != null);
            if (AuthCheck(AuthorizerAC.TRANSACTION, "ROLLBACK", null, null) > 0)
                return;
            var v = GetVdbe();
            if (v != null)
                v.AddOp2(OP_AutoCommit, 1, 1);
        }

        #endregion
    }
}
