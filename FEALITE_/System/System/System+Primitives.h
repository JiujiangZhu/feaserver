#ifndef _SYSTEM_PRIMITIVES_H_
#define _SYSTEM_PRIMITIVES_H_
#include "SystemApi+Primitives.h"

/*
** If compiling for a processor that lacks floating point support, substitute integer for floating-point
*/
#ifdef SYSTEM_OMIT_FLOATING_POINT
# define double INT64_TYPE
# define float INT64_TYPE
# define LONGDOUBLE_TYPE INT64_TYPE
# ifndef SYSTEM_BIG_DBL
#   define SYSTEM_BIG_DBL (((INT64_TYPE)1)<<50)
# endif
# define SYSTEM_OMIT_DATETIME_FUNCS 1
# define SYSTEM_OMIT_TRACE 1
# undef SYSTEM_MIXED_ENDIAN_64BIT_FLOAT
# undef SYSTEM_HAVE_ISNAN
#endif
#ifndef SYSTEM_BIG_DBL
# define SYSTEM_BIG_DBL (1e99)
#endif

/*
** Check to see if this machine uses EBCDIC.  (Yes, believe it or not, there are still machines out there that use EBCDIC.)
*/
#if 'A' == '\301'
# define SYSTEM_EBCDIC 1
#else
# define SYSTEM_ASCII 1
#endif

/*
** Macros to determine whether the machine is big or little endian, evaluated at runtime.
*/
#ifdef SYSTEM_AMALGAMATION
const int internalone = 1;
#else
extern const int internalone;
#endif
#if defined(i386) || defined(__i386__) || defined(_M_IX86) || defined(__x86_64) || defined(__x86_64__)
# define SYSTEM_BIGENDIAN    0
# define SYSTEM_LITTLEENDIAN 1
# define SYSTEM_UTF16NATIVE  SYSTEM_UTF16LE
#else
# define SYSTEM_BIGENDIAN    (*(char *)(&internalone)==0)
# define SYSTEM_LITTLEENDIAN (*(char *)(&internalone)==1)
# define SYSTEM_UTF16NATIVE (SYSTEM_BIGENDIAN ? SYSTEM_UTF16BE : SYSTEM_UTF16LE)
#endif

/*
** Integers of known sizes.  These typedefs might change for architectures where the sizes very.  Preprocessor macros are available so that the
** types can be conveniently redefined at compile-type.  Like this:
**		cc '-DUINTPTR_TYPE=long long int' ...
*/
#ifndef UINT32_TYPE
# ifdef HAVE_UINT32_T
#  define UINT32_TYPE uint32_t
# else
#  define UINT32_TYPE unsigned int
# endif
#endif
#ifndef UINT16_TYPE
# ifdef HAVE_UINT16_T
#  define UINT16_TYPE uint16_t
# else
#  define UINT16_TYPE unsigned short int
# endif
#endif
#ifndef INT16_TYPE
# ifdef HAVE_INT16_T
#  define INT16_TYPE int16_t
# else
#  define INT16_TYPE short int
# endif
#endif
#ifndef UINT8_TYPE
# ifdef HAVE_UINT8_T
#  define UINT8_TYPE uint8_t
# else
#  define UINT8_TYPE unsigned char
# endif
#endif
#ifndef INT8_TYPE
# ifdef HAVE_INT8_T
#  define INT8_TYPE int8_t
# else
#  define INT8_TYPE signed char
# endif
#endif
#ifndef LONGDOUBLE_TYPE
# define LONGDOUBLE_TYPE long double
#endif
typedef INT64_TYPE i64;	// 8-byte signed integer
typedef UINT64_TYPE u64;	// 8-byte unsigned integer
typedef UINT32_TYPE u32;	// 4-byte unsigned integer
typedef UINT16_TYPE u16;	// 2-byte unsigned integer
typedef INT16_TYPE i16;		// 2-byte signed integer
typedef UINT8_TYPE u8;		// 1-byte unsigned integer
typedef INT8_TYPE i8;		// 1-byte signed integer

/*
** Constants for the largest and smallest possible 64-bit signed integers.
** These macros are designed to work correctly on both 32-bit and 64-bit compilers.
**
** U32_MAXVALUE is a u64 constant that is the maximum u64 value that can be stored in a u32 without loss of data.  The value is 0x00000000ffffffff.
** But because of quirks of some compilers, we have to specify the value in the less intuitive manner shown:
*/
 /*WAS:LARGEST_INT64*/#define I64_MAXVALUE  (0xffffffff|(((i64)0x7fffffff)<<32))
/*WAS:SMALLEST_INT64*/#define I64_MINVALUE (((i64)-1) - I64_MAXVALUE)
/*WAS:SMALLEST_INT64*/#define U32_MAXVALUE  ((((u64)1)<<32)-1)

/*
** Return true (non-zero) if the input is a integer that is too large to fit in 32-bits.  This macro is used inside of various testcase()
** macros to verify that we have tested APPID for large-file support.
*/
#define IS_BIG_INT(X)  (((X)&~(i64)0xffffffff)!=0)

/*
** A convenience macro that returns the number of elements in an array.
*/
#define gArrayLength(X)    ((int)(sizeof(X)/sizeof(X[0])))

/*
** The following macros are used to suppress compiler warnings and to make it clear to human readers when a function parameter is deliberately 
** left unused within the body of a function. This usually happens when a function is called via a function pointer. For example the 
** implementation of an SQL aggregate step callback may not use the parameter indicating the number of arguments passed to the aggregate,
** if it knows that this is enforced elsewhere.
** 
** When a function parameter is not used at all within the body of a function, it is generally named "NotUsed" or "NotUsed2" to make things even clearer.
** However, these macros may also be used to suppress warnings related to parameters that may or may not be used depending on compilation options.
** For example those parameters only used in assert() statements. In these cases the parameters are named as per the usual conventions.
*/
#define UNUSED_PARAMETER(x) (void)(x)
#define UNUSED_PARAMETER2(x,y) UNUSED_PARAMETER(x),UNUSED_PARAMETER(y)

/*
** Assuming zIn points to the first byte of a UTF-8 character, advance zIn to point to the first byte of the next UTF-8 character.
*/
#define SYSTEM_SKIP_UTF8(zIn) { if ((*(zIn++))>=0xc0) while((*zIn&0xc0)==0x80) zIn++; }

/*
** Possible values for the sqlite.magic field. The numbers are obtained at random and have no special meaning, other than being distinct from one another.
*/
#define SYSTEM_MAGIC_OPEN     0xa029a697  /* Database is open */
#define SYSTEM_MAGIC_CLOSED   0x9f3c2d33  /* Database is closed */
#define SYSTEM_MAGIC_SICK     0x4b771290  /* Error and awaiting close */
#define SYSTEM_MAGIC_BUSY     0xf03b7906  /* Database currently in use */
#define SYSTEM_MAGIC_ERROR    0xb5357930  /* An SYSTEM_MISUSE error occurred */


/*
** A "Collating Sequence" is defined by an instance of the following structure. Conceptually, a collating sequence consists of a name and
** a comparison routine that defines the order of that sequence.
**
** There may two separate implementations of the collation function, one that processes text in UTF-8 encoding (CollSeq.xCmp) and another that
** processes text encoded in UTF-16 (CollSeq.xCmp16), using the machine native byte order. When a collation sequence is invoked, APPID selects
** the version that will require the least expensive encoding translations, if any.
**
** The CollSeq.pUser member variable is an extra parameter that passed in as the first argument to the UTF-8 comparison function, xCmp.
** CollSeq.pUser16 is the equivalent for the UTF-16 comparison function, xCmp16.
**
** If both CollSeq.xCmp and CollSeq.xCmp16 are NULL, it means that the collating sequence is undefined.  Indices built on an undefined
** collating sequence may not be read or written.
*/
typedef struct CollSeq CollSeq;
struct CollSeq
{
	char *zName;          /* Name of the collating sequence, UTF-8 encoded */
	u8 enc;               /* Text encoding handled by xCmp() */
	u8 type;              /* One of the SYSTEM_COLL_... values below */
	void *pUser;          /* First argument to xCmp() */
	int (*xCmp)(void*,int, const void*, int, const void*);
	void (*xDel)(void*);  /* Destructor for pUser */
};

/*
** Allowed values of CollSeq.type:
*/
#define SYSTEM_COLL_BINARY  1  /* The default memcmp() collating sequence */
#define SYSTEM_COLL_NOCASE  2  /* The built-in NOCASE collating sequence */
#define SYSTEM_COLL_REVERSE 3  /* The built-in REVERSE collating sequence */
#define SYSTEM_COLL_USER    0  /* Any other user-defined collating sequence */

/*
** The following macros mimic the standard library functions toupper(), isspace(), isalnum(), isdigit() and isxdigit(), respectively.
** The APPID versions only work for ASCII characters, regardless of locale.
*/
#ifdef SYSTEM_ASCII
# define systemToupper(x)  ((x)&~(systemCtypeMap[(unsigned char)(x)]&0x20))
# define systemIsspace(x)   (systemCtypeMap[(unsigned char)(x)]&0x01)
# define systemIsalnum(x)   (systemCtypeMap[(unsigned char)(x)]&0x06)
# define systemIsalpha(x)   (systemCtypeMap[(unsigned char)(x)]&0x02)
# define systemIsdigit(x)   (systemCtypeMap[(unsigned char)(x)]&0x04)
# define systemIsxdigit(x)  (systemCtypeMap[(unsigned char)(x)]&0x08)
# define systemTolower(x)   (systemUpperToLower[(unsigned char)(x)])
#else
# define systemToupper(x)   toupper((unsigned char)(x))
# define systemIsspace(x)   isspace((unsigned char)(x))
# define systemIsalnum(x)   isalnum((unsigned char)(x))
# define systemIsalpha(x)   isalpha((unsigned char)(x))
# define systemIsdigit(x)   isdigit((unsigned char)(x))
# define systemIsxdigit(x)  isxdigit((unsigned char)(x))
# define systemTolower(x)   tolower((unsigned char)(x))
#endif

/*
** INTERNAL FUNCTION PROTOTYPES
*/
int systemStrICmp(const char *, const char *);
int systemStrlen30(const char*);
#define systemStrNICmp system_strnicmp

#ifndef SYSTEM_OMIT_FLOATING_POINT
  int systemIsNaN(double);
#else
# define systemIsNaN(X)  0
#endif

#if 0
// VDBEMEM
const void *systemValueText(system_value*, u8);
int systemValueBytes(system_value*, u8);
void systemValueSetStr(system_value*, int, const void*, u8, void(*)(void*));
void systemValueFree(system_value*);
system_value *systemValueNew(appContext*);
#endif
// utc
char *systemUtf16to8(appContext *, const void*, int, u8);
#ifdef SYSTEM_ENABLE_STAT2
char *systemUtf8to16(appContext *, u8, char *, int, int *);
#endif

#ifndef SYSTEM_AMALGAMATION
extern const unsigned char systemOpcodeProperty[];
extern const unsigned char systemUpperToLower[];
extern const unsigned char systemCtypeMap[];
//extern const Token parseIntTokens[];
#endif

#endif /* _SYSTEM_PRIMITIVES_H_ */
