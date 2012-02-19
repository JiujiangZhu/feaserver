/*
** This file implements routines used to report what compile-time options APPID was built with.
*/
#ifndef SYSTEM_OMIT_COMPILEOPTION_DIAGS
#include "System.h"

// These macros are provided to "stringify" the value of the define for those options in which the value is meaningful.
#define CTIMEOPT_VAL_(opt) #opt
#define CTIMEOPT_VAL(opt) CTIMEOPT_VAL_(opt)

/*
** An array of names of all compile-time options.  This array should  be sorted A-Z.
**
** This array looks large, but in a typical installation actually uses only a handful of compile-time options, so most times this array is usually
** rather short and uses little memory space.
*/
static const char * const azCompileOpt[] = {
#ifdef SYSTEM_32BIT_ROWID
	"32BIT_ROWID",
#endif
#ifdef SYSTEM_4_BYTE_ALIGNED_MALLOC
	"4_BYTE_ALIGNED_MALLOC",
#endif
#ifdef SYSTEM_CASE_SENSITIVE_LIKE
	"CASE_SENSITIVE_LIKE",
#endif
#ifdef SYSTEM_CHECK_PAGES
	"CHECK_PAGES",
#endif
#ifdef SYSTEM_COVERAGE_TEST
	"COVERAGE_TEST",
#endif
#ifdef SYSTEM_DEBUG
	"DEBUG",
#endif
#ifdef SYSTEM_DEFAULT_LOCKING_MODE
	"DEFAULT_LOCKING_MODE=" CTIMEOPT_VAL(SYSTEM_DEFAULT_LOCKING_MODE),
#endif
#ifdef SYSTEM_DISABLE_DIRSYNC
	"DISABLE_DIRSYNC",
#endif
#ifdef SYSTEM_DISABLE_LFS
	"DISABLE_LFS",
#endif
#ifdef SYSTEM_ENABLE_ATOMIC_WRITE
	"ENABLE_ATOMIC_WRITE",
#endif
#ifdef SYSTEM_ENABLE_CEROD
	"ENABLE_CEROD",
#endif
#ifdef SYSTEM_ENABLE_COLUMN_METADATA
	"ENABLE_COLUMN_METADATA",
#endif
#ifdef SYSTEM_ENABLE_EXPENSIVE_ASSERT
	"ENABLE_EXPENSIVE_ASSERT",
#endif
#ifdef SYSTEM_ENABLE_FTS1
	"ENABLE_FTS1",
#endif
#ifdef SYSTEM_ENABLE_FTS2
	"ENABLE_FTS2",
#endif
#ifdef SYSTEM_ENABLE_FTS3
	"ENABLE_FTS3",
#endif
#ifdef SYSTEM_ENABLE_FTS3_PARENTHESIS
	"ENABLE_FTS3_PARENTHESIS",
#endif
#ifdef SYSTEM_ENABLE_FTS4
	"ENABLE_FTS4",
#endif
#ifdef SYSTEM_ENABLE_ICU
	"ENABLE_ICU",
#endif
#ifdef SYSTEM_ENABLE_IOTRACE
	"ENABLE_IOTRACE",
#endif
#ifdef SYSTEM_ENABLE_LOAD_EXTENSION
	"ENABLE_LOAD_EXTENSION",
#endif
#ifdef SYSTEM_ENABLE_LOCKING_STYLE
	"ENABLE_LOCKING_STYLE=" CTIMEOPT_VAL(SYSTEM_ENABLE_LOCKING_STYLE),
#endif
#ifdef SYSTEM_ENABLE_MEMORY_MANAGEMENT
	"ENABLE_MEMORY_MANAGEMENT",
#endif
#ifdef SYSTEM_ENABLE_MEMSYS3
	"ENABLE_MEMSYS3",
#endif
#ifdef SYSTEM_ENABLE_MEMSYS5
	"ENABLE_MEMSYS5",
#endif
#ifdef SYSTEM_ENABLE_OVERSIZE_CELL_CHECK
	"ENABLE_OVERSIZE_CELL_CHECK",
#endif
#ifdef SYSTEM_ENABLE_RTREE
	"ENABLE_RTREE",
#endif
#ifdef SYSTEM_ENABLE_STAT2
	"ENABLE_STAT2",
#endif
#ifdef SYSTEM_ENABLE_UNLOCK_NOTIFY
	"ENABLE_UNLOCK_NOTIFY",
#endif
#ifdef SYSTEM_ENABLE_UPDATE_DELETE_LIMIT
	"ENABLE_UPDATE_DELETE_LIMIT",
#endif
#ifdef SYSTEM_HAS_CODEC
	"HAS_CODEC",
#endif
#ifdef SYSTEM_HAVE_ISNAN
	"HAVE_ISNAN",
#endif
#ifdef SYSTEM_HOMEGROWN_RECURSIVE_MUTEX
	"HOMEGROWN_RECURSIVE_MUTEX",
#endif
#ifdef SYSTEM_IGNORE_AFP_LOCK_ERRORS
	"IGNORE_AFP_LOCK_ERRORS",
#endif
#ifdef SYSTEM_IGNORE_FLOCK_LOCK_ERRORS
	"IGNORE_FLOCK_LOCK_ERRORS",
#endif
#ifdef SYSTEM_INT64_TYPE
	"INT64_TYPE",
#endif
#ifdef SYSTEM_LOCK_TRACE
	"LOCK_TRACE",
#endif
#ifdef SYSTEM_MEMDEBUG
	"MEMDEBUG",
#endif
#ifdef SYSTEM_MIXED_ENDIAN_64BIT_FLOAT
	"MIXED_ENDIAN_64BIT_FLOAT",
#endif
#ifdef SYSTEM_NO_SYNC
	"NO_SYNC",
#endif
#ifdef SYSTEM_OMIT_ALTERTABLE
	"OMIT_ALTERTABLE",
#endif
#ifdef SYSTEM_OMIT_ANALYZE
	"OMIT_ANALYZE",
#endif
#ifdef SYSTEM_OMIT_ATTACH
	"OMIT_ATTACH",
#endif
#ifdef SYSTEM_OMIT_AUTHORIZATION
	"OMIT_AUTHORIZATION",
#endif
#ifdef SYSTEM_OMIT_AUTOINCREMENT
	"OMIT_AUTOINCREMENT",
#endif
#ifdef SYSTEM_OMIT_AUTOINIT
	"OMIT_AUTOINIT",
#endif
#ifdef SYSTEM_OMIT_AUTOMATIC_INDEX
	"OMIT_AUTOMATIC_INDEX",
#endif
#ifdef SYSTEM_OMIT_AUTOVACUUM
	"OMIT_AUTOVACUUM",
#endif
#ifdef SYSTEM_OMIT_BETWEEN_OPTIMIZATION
	"OMIT_BETWEEN_OPTIMIZATION",
#endif
#ifdef SYSTEM_OMIT_BLOB_LITERAL
	"OMIT_BLOB_LITERAL",
#endif
#ifdef SYSTEM_OMIT_BTREECOUNT
	"OMIT_BTREECOUNT",
#endif
#ifdef SYSTEM_OMIT_BUILTIN_TEST
	"OMIT_BUILTIN_TEST",
#endif
#ifdef SYSTEM_OMIT_CAST
	"OMIT_CAST",
#endif
#ifdef SYSTEM_OMIT_CHECK
	"OMIT_CHECK",
#endif
/* // redundant
** #ifdef SYSTEM_OMIT_COMPILEOPTION_DIAGS
**   "OMIT_COMPILEOPTION_DIAGS",
** #endif
*/
#ifdef SYSTEM_OMIT_COMPLETE
	"OMIT_COMPLETE",
#endif
#ifdef SYSTEM_OMIT_COMPOUND_SELECT
	"OMIT_COMPOUND_SELECT",
#endif
#ifdef SYSTEM_OMIT_DATETIME_FUNCS
	"OMIT_DATETIME_FUNCS",
#endif
#ifdef SYSTEM_OMIT_DECLTYPE
	"OMIT_DECLTYPE",
#endif
#ifdef SYSTEM_OMIT_DEPRECATED
	"OMIT_DEPRECATED",
#endif
#ifdef SYSTEM_OMIT_DISKIO
	"OMIT_DISKIO",
#endif
#ifdef SYSTEM_OMIT_EXPLAIN
	"OMIT_EXPLAIN",
#endif
#ifdef SYSTEM_OMIT_FLAG_PRAGMAS
	"OMIT_FLAG_PRAGMAS",
#endif
#ifdef SYSTEM_OMIT_FLOATING_POINT
	"OMIT_FLOATING_POINT",
#endif
#ifdef SYSTEM_OMIT_FOREIGN_KEY
	"OMIT_FOREIGN_KEY",
#endif
#ifdef SYSTEM_OMIT_GET_TABLE
	"OMIT_GET_TABLE",
#endif
#ifdef SYSTEM_OMIT_INCRBLOB
	"OMIT_INCRBLOB",
#endif
#ifdef SYSTEM_OMIT_INTEGRITY_CHECK
	"OMIT_INTEGRITY_CHECK",
#endif
#ifdef SYSTEM_OMIT_LIKE_OPTIMIZATION
	"OMIT_LIKE_OPTIMIZATION",
#endif
#ifdef SYSTEM_OMIT_LOAD_EXTENSION
	"OMIT_LOAD_EXTENSION",
#endif
#ifdef SYSTEM_OMIT_LOCALTIME
	"OMIT_LOCALTIME",
#endif
#ifdef SYSTEM_OMIT_LOOKASIDE
	"OMIT_LOOKASIDE",
#endif
#ifdef SYSTEM_OMIT_MEMORYDB
	"OMIT_MEMORYDB",
#endif
#ifdef SYSTEM_OMIT_OR_OPTIMIZATION
	"OMIT_OR_OPTIMIZATION",
#endif
#ifdef SYSTEM_OMIT_PAGER_PRAGMAS
	"OMIT_PAGER_PRAGMAS",
#endif
#ifdef SYSTEM_OMIT_PRAGMA
	"OMIT_PRAGMA",
#endif
#ifdef SYSTEM_OMIT_PROGRESS_CALLBACK
	"OMIT_PROGRESS_CALLBACK",
#endif
#ifdef SYSTEM_OMIT_QUICKBALANCE
	"OMIT_QUICKBALANCE",
#endif
#ifdef SYSTEM_OMIT_REINDEX
	"OMIT_REINDEX",
#endif
#ifdef SYSTEM_OMIT_SCHEMA_PRAGMAS
	"OMIT_SCHEMA_PRAGMAS",
#endif
#ifdef SYSTEM_OMIT_SCHEMA_VERSION_PRAGMAS
	"OMIT_SCHEMA_VERSION_PRAGMAS",
#endif
#ifdef SYSTEM_OMIT_SHARED_CACHE
	"OMIT_SHARED_CACHE",
#endif
#ifdef SYSTEM_OMIT_SUBQUERY
	"OMIT_SUBQUERY",
#endif
#ifdef SYSTEM_OMIT_TCL_VARIABLE
	"OMIT_TCL_VARIABLE",
#endif
#ifdef SYSTEM_OMIT_TEMPDB
	"OMIT_TEMPDB",
#endif
#ifdef SYSTEM_OMIT_TRACE
	"OMIT_TRACE",
#endif
#ifdef SYSTEM_OMIT_TRIGGER
	"OMIT_TRIGGER",
#endif
#ifdef SYSTEM_OMIT_TRUNCATE_OPTIMIZATION
	"OMIT_TRUNCATE_OPTIMIZATION",
#endif
#ifdef SYSTEM_OMIT_UTF16
	"OMIT_UTF16",
#endif
#ifdef SYSTEM_OMIT_VACUUM
	"OMIT_VACUUM",
#endif
#ifdef SYSTEM_OMIT_VIEW
	"OMIT_VIEW",
#endif
#ifdef SYSTEM_OMIT_VIRTUALTABLE
	"OMIT_VIRTUALTABLE",
#endif
#ifdef SYSTEM_OMIT_WAL
	"OMIT_WAL",
#endif
#ifdef SYSTEM_OMIT_WSD
	"OMIT_WSD",
#endif
#ifdef SYSTEM_OMIT_XFER_OPT
	"OMIT_XFER_OPT",
#endif
#ifdef SYSTEM_PERFORMANCE_TRACE
	"PERFORMANCE_TRACE",
#endif
#ifdef SYSTEM_PROXY_DEBUG
	"PROXY_DEBUG",
#endif
#ifdef SYSTEM_SECURE_DELETE
	"SECURE_DELETE",
#endif
#ifdef SYSTEM_SMALL_STACK
	"SMALL_STACK",
#endif
#ifdef SYSTEM_SOUNDEX
	"SOUNDEX",
#endif
#ifdef SYSTEM_TCL
	"TCL",
#endif
#ifdef SYSTEM_TEMP_STORE
	"TEMP_STORE=" CTIMEOPT_VAL(SYSTEM_TEMP_STORE),
#endif
#ifdef SYSTEM_TEST
	"TEST",
#endif
#ifdef SYSTEM_THREADSAFE
	"THREADSAFE=" CTIMEOPT_VAL(SYSTEM_THREADSAFE),
#endif
#ifdef SYSTEM_USE_ALLOCA
	"USE_ALLOCA",
#endif
#ifdef SYSTEM_ZERO_MALLOC
	"ZERO_MALLOC"
#endif
};

/*
** Given the name of a compile-time option, return true if that option was used and false if not.
** The name can optionally begin with "SYSTEM_" but the "SYSTEM_" prefix is not required for a match.
*/
int system_GetHasCompileOption(const char *zOptName)
{
	int i, n;
	if (systemStrNICmp(zOptName, "SYSTEM_", 7)==0)
		zOptName += 7;
	n = systemStrlen30(zOptName);
	/* Since gArrayLength(azCompileOpt) is normally in single digits, a linear search is adequate.  No need for a binary search. */
	for(i=0; i<gArrayLength(azCompileOpt); i++)
		if ((systemStrNICmp(zOptName, azCompileOpt[i], n)==0) && ((azCompileOpt[i][n]==0) || (azCompileOpt[i][n]=='=')))
			return 1;
	return 0;
}

/*
** Return the N-th compile-time option string.  If N is out of range, return a NULL pointer.
*/
const char *system_GetCompileOptionByID(int id)
{
	if (id>=0 && id<gArrayLength(azCompileOpt))
		return azCompileOpt[id];
	return 0;
}

#endif /*  SYSTEM_OMIT_COMPILEOPTION_DIAGS */
