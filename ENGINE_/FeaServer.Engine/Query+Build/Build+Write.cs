namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // Generate VDBE code that prepares for doing an operation that might change the database.
        //
        // This routine starts a new transaction if we are not already within a transaction.  If we are already within a transaction, then a checkpoint
        // is set if the setStatement parameter is true.  A checkpoint should be set for operations that might fail (due to a constraint) part of
        // the way through and which will need to undo some writes without having to rollback the whole transaction.  For operations where all constraints
        // can be checked before any changes are made to the database, it is never necessary to undo a write and the checkpoint should not be set.
        internal static void sqlite3BeginWriteOperation(Parse pParse, int setStatement, int iDb)
        {
            var pToplevel = sqlite3ParseToplevel(pParse);
            sqlite3CodeVerifySchema(pParse, iDb);
            pToplevel.writeMask |= ((yDbMask)1) << iDb;
            pToplevel.isMultiWrite |= (byte)setStatement;
        }

        // Indicate that the statement currently under construction might write more than one entry (example: deleting one row then inserting another,
        // inserting multiple rows in a table, or inserting a row and index entries.) If an abort occurs after some of these writes have completed, then it will
        // be necessary to undo the completed writes.
        internal static void sqlite3MultiWrite(Parse pParse)
        {
            var pToplevel = sqlite3ParseToplevel(pParse);
            pToplevel.isMultiWrite = 1;
        }

        // The code generator calls this routine if is discovers that it is possible to abort a statement prior to completion.  In order to 
        // perform this abort without corrupting the database, we need to make sure that the statement is protected by a statement transaction.
        //
        // Technically, we only need to set the mayAbort flag if the isMultiWrite flag was previously set.  There is a time dependency
        // such that the abort must occur after the multiwrite.  This makes some statements involving the REPLACE conflict resolution algorithm
        // go a little faster.  But taking advantage of this time dependency makes it more difficult to prove that the code is correct (in 
        // particular, it prevents us from writing an effective implementation of sqlite3AssertMayAbort()) and so we have chosen
        // to take the safe route and skip the optimization.
        internal static void sqlite3MayAbort(Parse pParse)
        {
            var pToplevel = sqlite3ParseToplevel(pParse);
            pToplevel.mayAbort = 1;
        }
    }
}