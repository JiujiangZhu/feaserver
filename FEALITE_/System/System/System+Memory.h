#ifndef _SYSTEM_MEMORY_H_
#define _SYSTEM_MEMORY_H_
#include "SystemApi+Memory.h"

/*
** The SYSTEM_DEFAULT_MEMSTATUS macro must be defined as either 0 or 1.
** It determines whether or not the features related to APPID_CONFIG_MEMSTATUS are available by default or not.
** This value can be overridden at runtime using the APPID_config() API.
*/
#if !defined(SYSTEM_DEFAULT_MEMSTATUS)
# define SYSTEM_DEFAULT_MEMSTATUS 1
#endif

/*
** Exactly one of the following macros must be defined in order to specify which memory allocation subsystem to use.
**		SYSTEM_MALLOC	// Use normal system malloc()
**      SYSTEM_DMALLOC	// Debugging version of system malloc()
** If none of the above are defined, then set SYSTEM_MALLOC as the default.
*/
#if defined(SYSTEM_MALLOC)+defined(SYSTEM_DMALLOC)>1
# error "At most one of the following compile-time configuration options is allows: SYSTEM_MALLOC, SYSTEM_DMALLOC"
#endif
#if defined(SYSTEM_MALLOC)+defined(SYSTEM_DMALLOC)==0
# define SYSTEM_MALLOC 1
#endif

/*
** If SYSTEM_MALLOC_SOFTLIMIT is not zero, then try to keep the sizes of memory allocations below this value where possible.
*/
#if !defined(SYSTEM_MALLOC_SOFTLIMIT)
# define SYSTEM_MALLOC_SOFTLIMIT 1024
#endif

/*
** Assert that the pointer X is aligned to an 8-byte boundary.  This macro is used only within assert() to verify that the code gets
** all alignment restrictions correct.
** 
** Except, if SYSTEM_4_BYTE_ALIGNED_MALLOC is defined, then the underlying malloc() implemention might return us 4-byte aligned
** pointers.  In that case, only verify 4-byte alignment.
*/
#ifdef SYSTEM_4_BYTE_ALIGNED_MALLOC
# define EIGHT_BYTE_ALIGNMENT(X)   ((((char*)(X) - (char*)0)&3)==0)
#else
# define EIGHT_BYTE_ALIGNMENT(X)   ((((char*)(X) - (char*)0)&7)==0)
#endif

/* 
** Round up a number to the next larger multiple of 8. This is used to force 8-byte alignment on 64-bit architectures.
*/
#define ROUND8(x)     (((x)+7)&~7)

/*
** Round down to the nearest multiple of 8
*/
#define ROUNDDOWN8(x) ((x)&~7)

/*
** Assert that the pointer X is aligned to an 8-byte boundary.  This macro is used only within assert() to verify that the code gets
** all alignment restrictions correct.
** 
** Except, if SYSTEM_4_BYTE_ALIGNED_MALLOC is defined, then the underlying malloc() implemention might return us 4-byte aligned
** pointers.  In that case, only verify 4-byte alignment.
*/
#ifdef SYSTEM_4_BYTE_ALIGNED_MALLOC
# define EIGHT_BYTE_ALIGNMENT(X)   ((((char*)(X) - (char*)0)&3)==0)
#else
# define EIGHT_BYTE_ALIGNMENT(X)   ((((char*)(X) - (char*)0)&7)==0)
#endif

/*
** Lookaside malloc is a set of fixed-size buffers that can be used to satisfy small transient memory allocation requests for objects
** associated with a particular database connection.  The use of lookaside malloc provides a significant performance enhancement
** (approx 10%) by avoiding numerous malloc/free requests while parsing SQL statements.
**
** The Lookaside structure holds configuration information about the lookaside malloc subsystem.  Each available memory allocation in
** the lookaside subsystem is stored on a linked list of LookasideSlot objects.
**
** Lookaside allocations are only allowed for objects that are associated with a particular database connection.  Hence, schema information cannot
** be stored in lookaside because in shared cache mode the schema information is shared by multiple database connections.  Therefore, while parsing
** schema information, the Lookaside.bEnabled flag is cleared so that lookaside allocations are not used to construct the schema objects.
*/
typedef struct Lookaside Lookaside;
typedef struct LookasideSlot LookasideSlot;
struct Lookaside
{
	u16 sz;                 /* Size of each buffer in bytes */
	u8 bEnabled;            /* False to disable new lookaside allocations */
	u8 bMalloced;           /* True if pStart obtained from system_malloc() */
	int nOut;               /* Number of buffers currently checked out */
	int mxOut;              /* Highwater mark for nOut */
	LookasideSlot *pFree;   /* List of available buffers */
	void *pStart;           /* First byte of available memory space */
	void *pEnd;             /* First byte past end of available space */
};
struct LookasideSlot {
	LookasideSlot *pNext;    /* Next buffer in the list of free buffers */
};

/*
** INTERNAL FUNCTION PROTOTYPES
*/
int systemMallocInit(void);
void systemMallocEnd(void);
void *systemMalloc(int);
void *systemMallocZero(int);
void *systemCtxMallocZero(appContext*, int);
void *systemCtxMallocRaw(appContext*, int);
char *systemCtxStrDup(appContext*,const char*);
char *systemCtxStrNDup(appContext*,const char*, int);
void *systemRealloc(void*, int);
void *systemCtxReallocOrFree(appContext*, void *, int);
void *systemCtxRealloc(appContext*, void *, int);
void systemCtxFree(appContext*, void*);
int systemMallocSize(void*);
int systemCtxMallocSize(appContext*, void*);
void *systemScratchMalloc(int);
void systemScratchFree(void*);
void *systemPageMalloc(int);
void systemPageFree(void*);
void systemMemSetDefault(void);
void systemBenignMallocHooks(void (*)(void), void (*)(void));
int systemHeapNearlyFull(void);

/*
** On systems with ample stack space and that support alloca(), make use of alloca() to obtain space for large automatic objects.
** By default, obtain space from malloc().
** 
** The alloca() routine never returns NULL.  This will cause code paths that deal with sqlite3StackAlloc() failures to be unreachable.
*/
#ifdef SYSTEM_USE_ALLOCA
# define systemStackAllocRaw(D,N)   alloca(N)
# define systemStackAllocZero(D,N)  memset(alloca(N), 0, N)
# define systemStackFree(D,P)       
#else
# define systemStackAllocRaw(D,N)   systemCtxMallocRaw(D,N)
# define systemStackAllocZero(D,N)  systemCtxMallocZero(D,N)
# define systemStackFree(D,P)       systemCtxFree(D,P)
#endif

#ifdef SYSTEM_ENABLE_MEMSYS3
const system_mem_methods *systemMemGetMemsys3(void);
#endif
#ifdef SYSTEM_ENABLE_MEMSYS5
const system_mem_methods *systemMemGetMemsys5(void);
#endif

/*
** These routines are available for the mem2.c debugging memory allocator only.  They are used to verify that different "types" of memory
** allocations are properly tracked by the system.
** 
** systemMemdebugSetType() sets the "type" of an allocation to one of the MEMTYPE_* macros defined below.  The type must be a bitmask with
** a single bit set.
** 
** systemMemdebugHasType() returns true if any of the bits in its second argument match the type set by the previous systemMemdebugSetType().
** systemMemdebugHasType() is intended for use inside assert() statements.
** 
** systemMemdebugNoType() returns true if none of the bits in its second argument match the type set by the previous sqlite3MemdebugSetType().
** 
** Perhaps the most important point is the difference between MEMTYPE_HEAP and MEMTYPE_LOOKASIDE.  If an allocation is MEMTYPE_LOOKASIDE, that means
** it might have been allocated by lookaside, except the allocation was too large or lookaside was already full.  It is important to verify
** that allocations that might have been satisfied by lookaside are not passed back to non-lookaside free() routines.  Asserts such as the
** example above are placed on the non-lookaside free() routines to verify this constraint. 
** 
** All of this is no-op for a production build.  It only comes into play when the SYSTEM_DALLOC compile-time option is used.
*/
#ifdef SYSTEM_DALLOC
void systemMemdebugSetType(void*,u8);
int systemMemdebugHasType(void*,u8);
int systemMemdebugNoType(void*,u8);
#else
# define systemMemdebugSetType(X,Y)  // no-op
# define systemMemdebugHasType(X,Y)  1
# define systemMemdebugNoType(X,Y)   1
#endif
#define MEMTYPE_HEAP       0x01  // General heap allocations
#define MEMTYPE_LOOKASIDE  0x02  // Might have been lookaside memory
#define MEMTYPE_SCRATCH    0x04  // Scratch allocations
#define MEMTYPE_PCACHE     0x08  // Page cache allocations
#define MEMTYPE_DB         0x10  // Uses sqlite3DbMalloc, not sqlite_malloc

/*
** The interface to the code in fault.c used for identifying "benign" malloc failures. This is only present if SQLITE_OMIT_BUILTIN_TEST
** is not defined.
*/
#ifndef SYSTEM_OMIT_BUILTIN_TEST
  void systemBeginBenignMalloc(void);
  void systemEndBenignMalloc(void);
#else
  #define systemBeginBenignMalloc()
  #define systemEndBenignMalloc()
#endif


/*
** The following value as a destructor means to use systemDbFree(). This is an internal extension to SQLITE_STATIC and SQLITE_TRANSIENT.
*/
#define SYSTEM_DYNAMIC  ((system_destructor_type)systemCtxFree)

#endif /* _SYSTEM_MEMORY_H_ */
