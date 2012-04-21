using System.Diagnostics;
using System;
using FeaServer.Core;
namespace FeaServer.Query
{
    public partial class Parse
    {
        private const int SQLITE_N_COLCACHE = 0;
        private const int SQLITE_MAX_ATTACHED = 0;
        
        private struct ColCache
        {
            public int iTable;      /* Table cursor number */
            public int iColumn;     /* Table column number */
            public byte tempReg;    /* iReg is a temp register that needs to be freed */
            public int iLevel;      /* Nesting level */
            public int iReg;        /* Reg with value of this column. 0 means none. */
            public int lru;         /* Least recently used entry has the smallest value */
        }

        private Context _db;         /* The main database structure */
        private string _errorMsg;   /* An error message */
        private IVdbe _vbde;        /* An engine for executing database bytecode */
        private int _rc;            /* Return code from execution */
        private byte _colNamesSet;  /* TRUE after OP_ColumnName has been issued to pVdbe */
        private byte _checkSchema;  /* Causes schema cookie check after an error */
        private byte _nested;       /* Number of nested calls to the parser/code generator */
        private byte _nTempReg;     /* Number of temporary registers in aTempReg[] */
        private byte _nTempInUse;   /* Number of aTempReg[] currently checked out */
        private byte _nColCache;    /* Number of entries in aColCache[] */
        private byte _iColCache;    /* Next entry in aColCache[] to replace */
        private byte _isMultiWrite; /* True if statement may modify/insert multiple rows */
        private byte _mayAbort;     /* True if statement may throw an ABORT exception */
        private int[] _aTempReg = new int[8]; /* Holding area for temporary registers */
        private int _nRangeReg;     /* Size of the temporary register block */
        private int _iRangeReg;     /* First register in temporary register block */
        private int _nErr;          /* Number of errors seen */
        private int _nTab;          /* Number of previously allocated VDBE cursors */
        private int _nMem;          /* Number of memory cells used so far */
        private int _nSet;          /* Number of sets used so far */
        private int _nOnce;         /* Number of OP_Once instructions so far */
        private int _ckBase;        /* Base register of data during check constraints */
        private int _iCacheLevel;   /* ColCache valid when aColCache[].iLevel<=iCacheLevel */
        private int _iCacheCnt;     /* Counter used to generate aColCache[].lru values */
        private ColCache[] aColCache = new ColCache[SQLITE_N_COLCACHE]; /* One for each column cache entry */
        private uint _writeMask;   /* Start a write transaction on these databases */
        private uint _cookieMask;      /* Bitmask of schema verified databases */
        private int _cookieGoto;   /* Address of OP_Goto to cookie verifier subroutine */
        private int[] _cookieValue = new int[SQLITE_MAX_ATTACHED + 2];  /* Values of cookies to verify */
        private int _regRowid;        /* Register holding rowid of CREATE TABLE entry */
        private int _regRoot;         /* Register holding root page number for new objects */
        private int _nMaxArg;         /* Max args passed to user function by sub-program */
        private AutoincInfo _pAinc;  /* Information about AUTOINCREMENT counters */
        // Information used while coding trigger programs.
        private Parse _pToplevel;    /* Parse structure for main program (or NULL) */
        private Table _pTriggerTab;  /* Table triggers are being coded for */
        private double _nQueryLoop;   /* Estimated number of iterations of a query */
        private uint _oldmask;         /* Mask of old.* columns referenced */
        private uint _newmask;         /* Mask of new.* columns referenced */
        private byte _eTriggerOp;       /* TK_UPDATE, TK_INSERT or TK_DELETE */
        private byte _eOrconf;          /* Default ON CONFLICT policy for trigger steps */
        private byte _disableTriggers;  /* True to disable triggers */

        #region Above is constant between recursions.  Below is reset before and after each recursion

        private int _nVar;               // Number of '?' variables seen in the SQL so far
        private int _nzVar;              // Number of available slots in azVar[]
        private byte _explain;           // True if the EXPLAIN flag is found on the query
        private byte _declareVtab;       // True if inside sqlite3_declare_vtab()
        private int _nVtabLock;          // Number of virtual tables to lock
        private int _nAlias;             // Number of aliased result set columns
        private int _nHeight;            // Expression tree height of current sub-select
        private int _iSelectId;          // ID of current select for EXPLAIN output
        private int _iNextSelectId;      // Next available select ID for EXPLAIN output
        private string[] _azVar;         // Pointers to names of parameters
        private IVdbe _pReprepare;       // VM being reprepared (sqlite3Reprepare())
        private int[] _aAlias;           // Register used to hold aliased result
        private string _zTail;           // All SQL text past the last semicolon parsed
        private Table _pNewTable;        // A table being constructed by CREATE TABLE
        private Trigger _pNewTrigger;    // Trigger under construct by a CREATE TRIGGER
        private string _authContext;    // The 6th parameter to db->xAuth callbacks
        private Token _sNameToken;       // Token with unqualified schema object name
        private Token _sLastToken;       // The last token parsed
        private Token _sArg;             // Complete text of a module argument
        private Table[] _apVtabLock;     // Pointer to virtual tables needing locking
        private Table _pZombieTab;       // List of Table objects to delete after code gen
        private TriggerPrg _pTriggerPrg; // Linked list of coded triggers

        #endregion

        private object GetAndResetState() { return null; }
        private void RestoreState(object state) { }

        public void ErrorMsg(string format, params object[] args)
        {
            _errorMsg = string.Format(format, args);
            _rc = 0;
        }

        private IVdbe GetVdbe() { throw new NotImplementedException(); }

        //private void TableLock(int iDb, int iTab, byte isWriteLock, string zName) { }
        //private void codeTableLocks() { }
        //private Table LocateTable(int isView, string name, string dbase) { return new Table(); }
        //private void OpenMasterTable(int iDb) { }
    }
}
