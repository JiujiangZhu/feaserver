/*
** This file contains a no-op memory allocation drivers for use when SQLITE_ZERO_MALLOC is defined.  The allocation drivers implemented
** here always fail.  SQLite will not operate with these drivers.  These are merely placeholders.  Real drivers must be substituted using
** system_config() before APPID will operate.
*/
#include "System.h"

/*
** This version of the memory allocator is the default.  It is used when no other memory allocator is specified using compile-time
** macros.
*/
#ifdef SYSTEM_ZERO_MALLOC

/*
** No-op versions of all memory allocation routines
*/
static void *systemMemMalloc(int nByte) { return 0; }
static void systemMemFree(void *pPrior) { }
static void *systemMemRealloc(void *pPrior, int nByte) { return 0; }
static int systemMemSize(void *pPrior) { return 0; }
static int systemMemRoundup(int n) { return n; }
static int systemMemInit(void *NotUsed) { return SYSTEM_OK; }
static void systemMemShutdown(void *NotUsed) { }

/*
** This routine is the only routine in this file with external linkage.
**
** Populate the low-level memory allocation function pointers in systemGlobalConfig.m with pointers to the routines in this file.
*/
void systemMemSetDefault(void)
{
	static const system_mem_methods defaultMethods = {
		systemMemMalloc,
		systemMemFree,
		systemMemRealloc,
		systemMemSize,
		systemMemRoundup,
		systemMemInit,
		systemMemShutdown,
		0 };
	system_config(SYSTEM_CONFIG_MALLOC, &defaultMethods);
}

#endif /* SYSTEM_ZERO_MALLOC */
