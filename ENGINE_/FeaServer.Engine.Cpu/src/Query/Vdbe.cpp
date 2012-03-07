#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
#pragma once

namespace Query {

/* Invoke this macro on memory cells just prior to changing the value of the cell.  This macro verifies that shallow copies are not misused. */
#ifdef SQLITE_DEBUG
#define DEBUG_MemAboutToChange(P,M) sqlite3VdbeMemPrepareToChange(P,M)
#else
#define DEBUG_MemAboutToChange(P,M)
#endif

class Vdbe
{
private:
//	sqlite3* _ctx;            /* The database connection that owns this statement */
	Op* _ops;                /* Space to hold the virtual machine's program */
	Mem* _mems;              /* The memory locations */
//	Mem** _apArg;            /* Arguments to currently executing user function */
	Mem* _columnNames;          /* Column names to return */
//	Mem* _pResultSet;        /* Pointer to an array of results */
	int _memCount;               /* Number of memory locations currently allocated */
	int _opCount;                /* Number of instructions in the program */
//	int _nOpAlloc;           /* Number of slots allocated for aOp[] */
//	int _nLabel;             /* Number of labels used */
//	int _nLabelAlloc;        /* Number of slots allocated in aLabel[] */
//	int* _aLabel;            /* Space to hold the labels */
//	u16 _nResColumn;         /* Number of columns in one row of the result set */
//	u16 _nCursor;            /* Number of slots in apCsr[] */
//	u32 _magic;              /* Magic number for sanity checking */
//	char* _zErrMsg;          /* Error message written here */
//	Vdbe* _pPrev, *_pNext;     /* Linked list of VDBEs with the same Vdbe.db */
//	VdbeCursor **_apCsr;     /* One element of this array for each open cursor */
//	Mem *_aVar;              /* Values for the OP_Variable opcode. */
//	char **_azVar;           /* Name of variables */
//	ynVar _nVar;             /* Number of entries in aVar[] */
//	ynVar _nzVar;            /* Number of entries in azVar[] */
//	u32 _cacheCtr;           /* VdbeCursor row cache generation counter */
	int _pc;                 /* The program counter */
//	int _rc;                 /* Value to return */
//	u8 _errorAction;         /* Recovery action to do in case of an error */
//	u8 _explain;             /* True if EXPLAIN present on SQL command */
//	u8 _changeCntOn;         /* True to update the change-counter */
//	u8 _expired;             /* True if the VM needs to be recompiled */
//	u8 _runOnlyOnce;         /* Automatically expire on reset */
//	u8 _minWriteFileFormat;  /* Minimum file format for writable database files */
//	u8 _inVtabMethod;        /* See comments above */
//	u8 _usesStmtJournal;     /* True if uses a statement journal */
//	u8 _readOnly;            /* True for read-only statements */
//	u8 _isPrepareV2;         /* True if prepared with prepare_v2() */
//	int _nChange;            /* Number of db changes made since last reset */
//	yDbMask _btreeMask;      /* Bitmask of db->aDb[] entries referenced */
//	yDbMask _lockMask;       /* Subset of btreeMask that requires a lock */
//	int _iStatement;         /* Statement number (or 0 if has not opened stmt) */
//	int _aCounter[3];        /* Counters used by sqlite3_stmt_status() */
//#ifndef SQLITE_OMIT_TRACE
//	i64 _startTime;          /* Time when query started - used for profiling */
//#endif
//	i64 _nFkConstraint;      /* Number of imm. FK constraints this VM */
//	i64 _nStmtDefCons;       /* Number of def. constraints when stmt started */
//	char *_zSql;             /* Text of the SQL statement that generated this */
//	void *_pFree;            /* Free this when deleting the vdbe */
//#ifdef SQLITE_DEBUG
//	FILE *_trace;            /* Write an execution trace here, if not NULL */
//#endif
//	VdbeFrame *_pFrame;      /* Parent frame */
//	VdbeFrame *_pDelFrame;   /* List of frame objects to free on VM reset */
//	int _nFrame;             /* Number of frames in pFrame list */
//	u32 _expmask;            /* Binding to these vars invalidates VM */
//	SubProgram *_pProgram;   /* Linked list of all sub-programs used by VM */

public:
int VdbeExec()
{
//	Op *aOp = p->aOp;          /* Copy of p->aOp */
//	Op *pOp;                   /* Current operation */
//	int rc = SQLITE_OK;        /* Value to return */
//	sqlite3 *db = p->db;       /* The database */
//	u8 resetSchemaOnFault = 0; /* Reset schema after an error if positive */
//	u8 encoding = ENC(db);     /* The database encoding */
//
	//Mem *mem = p->aMem;       /* Copy of p->aMem */
	Mem *in1 = nullptr;             /* 1st input operand */
	Mem *in2 = nullptr;             /* 2nd input operand */
	Mem *in3 = nullptr;             /* 3rd input operand */
	Mem *out = nullptr;             /* Output operand */
//	int iCompare = 0;          /* Result of last OP_Compare operation */
//	int *aPermute = 0;         /* Permutation of columns for OP_Compare */
//	i64 lastRowid = db->lastRowid;  /* Saved value of the last insert ROWID */
//
//	ASSERT(_magic == VDBE_MAGIC_RUN);  /* sqlite3_step() verifies this */
//	sqlite3VdbeEnter(p);
//	if (_rc == SQLITE_NOMEM)
//	{
//		/* This happens if a malloc() inside a call to sqlite3_column_text() or sqlite3_column_text16() failed.  */
//		goto no_mem;
//	}
//	ASSERT((_rc == SQLITE_OK) || (_rc == SQLITE_BUSY));
//	_rc = SQLITE_OK;
//	ASSERT(_explain == 0);
//	_pResultSet = 0;
//	_db->busyHandler.nBusy = 0;
//	CHECK_FOR_INTERRUPT;
//	sqlite3VdbeIOTraceSql(p);
//#ifndef SQLITE_OMIT_PROGRESS_CALLBACK
//	int checkProgress = (db->xProgress != 0);/* True if progress callbacks are enabled */
//	int nProgressOps = 0;      /* Opcodes executed since progress callback. */
//#endif
//#ifdef SQLITE_DEBUG
//	sqlite3BeginBenignMalloc();
//	if ((_pc == 0) && ((_db->flags & SQLITE_VdbeListing) != 0))
//	{
//		printf("VDBE Program Listing:\n");
//		sqlite3VdbePrintSql(p);
//		for (int index = 0; index < _nOp; index++)
//			sqlite3VdbePrintOp(stdout, index, &aOp[index]);
//	}
//	sqlite3EndBenignMalloc();
//#endif

for (int pc = _pc; rc == SQLITE_OK; pc++) /* pc: The program counter */
{
	ASSERT((pc >= 0) && (pc < _opCount));
	if (_ctx->mallocFailed)
		goto no_mem;
#ifdef VDBE_PROFILE
	//int origPc = pc; /* Program counter at start of opcode */
	//u64 start = sqlite3Hwtime(); /* CPU clock count at start of opcode */
#endif
	Op *op = _ops[pc];
	/* Only allow tracing if SQLITE_DEBUG is defined. */
#ifdef SQLITE_DEBUG
	if (_trace)
	{
		if (pc == 0)
		{
			printf("VDBE Execution Trace:\n");
			sqlite3VdbePrintSql(p);
		}
		sqlite3VdbePrintOp(_trace, pc, pOp);
	}
#endif
      

#ifndef SQLITE_OMIT_PROGRESS_CALLBACK
    /* Call the progress callback if it is configured and the required number of VDBE ops have been executed (either since this invocation of
    ** sqlite3VdbeExec() or since last time the progress callback was called). If the progress callback returns non-zero, exit the virtual machine with a return code SQLITE_ABORT. */
	if (checkProgress)
	{
		if (_ctx->ProgressOpsCount == progressOpsCount)
		{
			int prc = _ctx->xProgress(db->pProgressArg);
			if (prc != 0)
			{
				rc = SQLITE_INTERRUPT;
				goto vdbe_error_halt;
			}
			progressOpsCount = 0;
		}
		progressOpsCount++;
	}
#endif

    /* On any opcode with the "out2-prerelase" tag, free any external allocations out of mem[p2] and set mem[p2] to be
    ** an undefined integer.  Opcodes will either fill in the integer value or convert mem[p2] to a different type. */
	ASSERT(op->opflags == sqlite3OpcodeProperty[op->opcode]);
    if (op->opflags & OPFLG_OUT2_PRERELEASE)
	{
		ASSERT(op->p2 > 0);
		ASSERT(op->p2 <= _memCount);
		pOut = &_mems[pOp->p2];
		DEBUG_MemAboutToChange(pOut);
		MemReleaseExt(pOut);
		pOut->flags = MEM_Int;
    }
	
	/* Sanity checking on other operands */
#ifdef SQLITE_DEBUG
	if ((op->opflags & OPFLG_IN1) != 0)
	{
		ASSERT(op->p1 > 0);
		ASSERT(op->p1 <= _memCount);
		ASSERT(memIsValid(&_mems[op->p1]));
		REGISTER_TRACE(op->p1, &_mems[op->p1]);
    }
    if ((op->opflags & OPFLG_IN2) != 0)
	{
		ASSERT(op->p2 > 0);
		ASSERT(op->p2 <= _memCount);
		ASSERT(memIsValid(_mems[op->p2]));
		REGISTER_TRACE(pOp->p2, &_mems[op->p2]);
    }
    if ((op->opflags & OPFLG_IN3) != 0)
	{
		ASSERT(op->p3 > 0);
		ASSERT(op->p3 <= _memCount);
		ASSERT(memIsValid(&_mems[op->p3]));
		REGISTER_TRACE(op->p3, &_mems[op->p3]);
    }
    if ((op->opflags & OPFLG_OUT2) != 0)
	{
		ASSERT(op->p2 > 0);
		ASSERT(op->p2 <= _memCount);
		DEBUG_MemAboutToChange(&_mems[op->p2]);
    }
    if ((op->opflags & OPFLG_OUT3) != 0)
	{
		ASSERT(op->p3 > 0);
		ASSERT(op->p3 <= _memCount);
		DEBUG_MemAboutToChange(&_mems[op->p3]);
    }
#endif
  
    switch (op->opcode)
	{
	}

}

}
}
}