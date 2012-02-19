/*
** This file contains low-level memory allocation drivers for when APPID will use the standard C-library malloc/realloc/free interface
** to obtain the memory it needs.
**
** This file contains implementations of the low-level memory allocation routines specified in the sqlite3_mem_methods object.
*/
#include "System.h"

/*
** This version of the memory allocator is the default.  It is used when no other memory allocator is specified using compile-time
** macros.
*/
#ifdef SYSTEM_SYSTEM_MALLOC

/*
** Like malloc(), but remember the size of the allocation so that we can find it later using systemMemSize().
**
** For this low-level routine, we are guaranteed that nByte>0 because cases of nByte<=0 will be intercepted and dealt with by higher level
** routines.
*/
static void *sqlite3MemMalloc(int nByte)
{
	i64 *p;
	assert(nByte > 0);
	nByte = ROUND8(nByte);
	p = malloc(nByte+8);
	if (p)
	{
		p[0] = nByte;
		p++;
	}
	else
	{
		testcase(systemGlobalConfig.xLog != 0);
		system_log(SYSTEM_NOMEM, "failed to allocate %u bytes of memory", nByte);
	}
	return (void *)p;
}

/*
** Like free() but works for allocations obtained from systemMemMalloc() or systemMemRealloc().
**
** For this low-level routine, we already know that pPrior!=0 since cases where pPrior==0 will have been intecepted and dealt with
** by higher-level routines.
*/
static void sqlite3MemFree(void *pPrior)
{
	i64 *p = (i64*)pPrior;
	assert(pPrior != 0);
	p--;
	free(p);
}

/*
** Report the allocated size of a prior return from xMalloc() or xRealloc().
*/
static int sqlite3MemSize(void *pPrior)
{
	i64 *p;
	if (pPrior == 0)
		return 0;
	p = (i64*)pPrior;
	p--;
	return (int)p[0];
}

/*
** Like realloc().  Resize an allocation previously obtained from systemMemMalloc().
**
** For this low-level interface, we know that pPrior!=0.  Cases where pPrior==0 while have been intercepted by higher-level routine and
** redirected to xMalloc.  Similarly, we know that nByte>0 becauses cases where nByte<=0 will have been intercepted by higher-level
** routines and redirected to xFree.
*/
static void *systemMemRealloc(void *pPrior, int nByte)
{
	i64 *p = (i64*)pPrior;
	assert(pPrior != 0 && nByte > 0);
	assert(nByte == ROUND8(nByte)); /* EV: R-46199-30249 */
	p--;
	p = realloc(p, nByte+8 );
	if( p ){
		p[0] = nByte;
		p++;
	}
	else
	{
	    testcase(systemGlobalConfig.xLog != 0);
		system_log(SYSTEM_NOMEM, "failed memory resize %u to %u bytes", systemMemSize(pPrior), nByte);
	}
	return (void*)p;
}

/*
** Round up a request size to the next valid allocation size.
*/
static int systemMemRoundup(int n)
{
	return ROUND8(n);
}

/*
** Initialize this module.
*/
static int systemMemInit(void *NotUsed)
{
	UNUSED_PARAMETER(NotUsed);
	return SYSTEM_OK;
}

/*
** Deinitialize this module.
*/
static void sqlite3MemShutdown(void *NotUsed){
  UNUSED_PARAMETER(NotUsed);
  return;
}

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

#endif /* SYSTEM_SYSTEM_MALLOC */
