/*
** This file contains definitions of global variables and contants.
*/
#include "System.h"

/* An array to map all upper-case characters into their corresponding lower-case character. 
**
** APPID only considers US-ASCII (or EBCDIC) characters.  We do not handle case conversions for the UTF character set since the tables
** involved are nearly as big or bigger than APPID itself.
*/
const unsigned char systemUpperToLower[] = {
#ifdef SYSTEM_ASCII
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
     36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
     54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 97, 98, 99,100,101,102,103,
    104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,
    122, 91, 92, 93, 94, 95, 96, 97, 98, 99,100,101,102,103,104,105,106,107,
    108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,
    126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,
    144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,
    162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,
    180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,
    198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,
    216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,
    234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,
    252,253,254,255
#endif
#ifdef SYSTEM_EBCDIC
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, /* 0x */
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, /* 1x */
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, /* 2x */
     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, /* 3x */
     64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, /* 4x */
     80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, /* 5x */
     96, 97, 66, 67, 68, 69, 70, 71, 72, 73,106,107,108,109,110,111, /* 6x */
    112, 81, 82, 83, 84, 85, 86, 87, 88, 89,122,123,124,125,126,127, /* 7x */
    128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143, /* 8x */
    144,145,146,147,148,149,150,151,152,153,154,155,156,157,156,159, /* 9x */
    160,161,162,163,164,165,166,167,168,169,170,171,140,141,142,175, /* Ax */
    176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191, /* Bx */
    192,129,130,131,132,133,134,135,136,137,202,203,204,205,206,207, /* Cx */
    208,145,146,147,148,149,150,151,152,153,218,219,220,221,222,223, /* Dx */
    224,225,162,163,164,165,166,167,168,169,232,203,204,205,206,207, /* Ex */
    239,240,241,242,243,244,245,246,247,248,249,219,220,221,222,255, /* Fx */
#endif
};

/*
** The following 256 byte lookup table is used to support APPID built-in equivalents to the following standard library functions:
**
**   isspace()                        0x01
**   isalpha()                        0x02
**   isdigit()                        0x04
**   isalnum()                        0x06
**   isxdigit()                       0x08
**   toupper()                        0x20
**   APPID identifier character       0x40
**
** Bit 0x20 is set if the mapped character requires translation to upper case. i.e. if the character is a lower-case ASCII character.
** If x is a lower-case ASCII character, then its upper-case equivalent is (x - 0x20). Therefore toupper() can be implemented as:
**
**   (x & ~(map[x]&0x20))
**
** Standard function tolower() is implemented using the systemUpperToLower[] array. tolower() is used more often than toupper() by APPID.
**
** Bit 0x40 is set if the character non-alphanumeric and can be used in an APPID identifier.  Identifiers are alphanumerics, "_", "$", and any
** non-ASCII UTF character. Hence the test for whether or not a character is part of an identifier is 0x46.
**
** APPID's versions are identical to the standard versions assuming a locale of "C". They are implemented as macros in System.h.
*/
#ifdef SYSTEM_ASCII
const unsigned char systemCtypeMap[256] = {
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 00..07    ........ */
	0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,  /* 08..0f    ........ */
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 10..17    ........ */
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 18..1f    ........ */
	0x01, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,  /* 20..27     !"#$%&' */
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 28..2f    ()*+,-./ */
	0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,  /* 30..37    01234567 */
	0x0c, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 38..3f    89:;<=>? */

	0x00, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x02,  /* 40..47    @ABCDEFG */
	0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,  /* 48..4f    HIJKLMNO */
	0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,  /* 50..57    PQRSTUVW */
	0x02, 0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x40,  /* 58..5f    XYZ[\]^_ */
	0x00, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x2a, 0x22,  /* 60..67    `abcdefg */
	0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22,  /* 68..6f    hijklmno */
	0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22,  /* 70..77    pqrstuvw */
	0x22, 0x22, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00,  /* 78..7f    xyz{|}~. */

	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 80..87    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 88..8f    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 90..97    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* 98..9f    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* a0..a7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* a8..af    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* b0..b7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* b8..bf    ........ */

	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* c0..c7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* c8..cf    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* d0..d7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* d8..df    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* e0..e7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* e8..ef    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  /* f0..f7    ........ */
	0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40   /* f8..ff    ........ */
};
#endif

/*
** The following singleton contains the global configuration for the APPID library.
*/
SYSTEM_WSD struct systemConfig systemConfig = {
	SYSTEM_DEFAULT_MEMSTATUS,  /* bMemstat */
	1,                         /* bCoreMutex */
	SYSTEM_THREADSAFE==1,      /* bFullMutex */
	0x7ffffffe,                /* mxStrlen */
	100,                       /* szLookaside */
	500,                       /* nLookaside */
	{0,0,0,0,0,0,0,0},         /* m */
	{0,0,0,0,0,0,0,0,0},       /* mutex */
	{0,0,0,0,0,0,0,0,0,0,0},   /* pcache */
	(void*)0,                  /* pHeap */
	0,                         /* nHeap */
	0, 0,                      /* mnHeap, mxHeap */
	(void*)0,                  /* pScratch */
	0,                         /* szScratch */
	0,                         /* nScratch */
	(void*)0,                  /* pPage */
	0,                         /* szPage */
	0,                         /* nPage */
	0,                         /* mxParserStack */
	0,                         /* sharedCacheEnabled */
	/* All the rest should always be initialized to zero */
	0,                         /* isInit */
	0,                         /* inProgress */
	0,                         /* isMutexInit */
	0,                         /* isMallocInit */
	0,                         /* isPCacheInit */
	0,                         /* pInitMutex */
	0,                         /* nRefInitMutex */
	0,                         /* xLog */
	0,                         /* pLogArg */
};

/*
** Hash table for global functions - functions common to all database connections.  After initialization, this table is read-only.
*/
//SYSTEM_WSD FuncDefHash systemGlobalFunctions;

/*
** Constant tokens for values 0 and 1.
*/
//const Token systemIntTokens[] = {
//	{ "0", 1 },
//	{ "1", 1 }
//};

/*
** The value of the "pending" byte must be 0x40000000 (1 byte past the 1-gibabyte boundary) in a compatible database.  APPID never uses
** the database page that contains the pending byte.  It never attempts to read or write that page.  The pending byte page is set assign
** for use by the VFS layers as space for managing file locks.
**
** During testing, it is often desirable to move the pending byte to a different position in the file.  This allows code that has to
** deal with the pending byte to run on files that are much smaller than 1 GiB.  The system_test_control() interface can be used to
** move the pending byte.
**
** IMPORTANT:  Changing the pending byte to any value other than 0x40000000 results in an incompatible database file format!
** Changing the pending byte during operating results in undefined and dileterious behavior.
*/
#ifndef SYSTEM_OMIT_WSD
int systemPendingByte = 0x40000000;
#endif

#include "Opcodes.h"
/*
** Properties of opcodes.  The OPFLG_INITIALIZER macro is created by mkopcodeh.awk during compilation.  Data is obtained
** from the comments following the "case OP_xxxx:" statements in the vdbe.c file.  
*/
const unsigned char systemOpcodeProperty[] = OPFLG_INITIALIZER;



/*
** Utility functions used throughout sqlite.
**
** This file contains functions for allocating memory, comparing strings, and stuff like that.
**
*/
#include <stdarg.h>
#ifdef SYSTEM_HAVE_ISNAN
# include <math.h>
#endif

/*
** Routine needed to support the testcase() macro.
*/
#ifdef SYSTEM_COVERAGE_TEST
void systemCoverage(int x)
{
	static int dummy = 0;
	dummy += x;
}
#endif

#ifndef SYSTEM_OMIT_FLOATING_POINT
/*
** Return true if the floating point value is Not a Number (NaN).
**
** Use the math library isnan() function if compiled with SYSTEM_HAVE_ISNAN. Otherwise, we have our own implementation that works on most systems.
*/
int systemIsNaN(double x)
{
	int rc;   /* The value return */
#if !defined(SYSTEM_HAVE_ISNAN)
	/* Systems that support the isnan() library function should probably make use of it by compiling with -DSQLITE_HAVE_ISNAN.  But we have
	** found that many systems do not have a working isnan() function so this implementation is provided as an alternative.
	**
	** This NaN test sometimes fails if compiled on GCC with -ffast-math. On the other hand, the use of -ffast-math comes with the following
	** warning:
	**
	**      This option [-ffast-math] should never be turned on by any -O option since it can result in incorrect output for programs
	**      which depend on an exact implementation of IEEE or ISO rules/specifications for math functions.
	**
	** Under MSVC, this NaN test may fail if compiled with a floating-point precision mode other than /fp:precise.  From the MSDN 
	** documentation:
	**
	**      The compiler [with /fp:precise] will properly handle comparisons involving NaN. For example, x != x evaluates to true if x is NaN 
	**      ... */
#ifdef __FAST_MATH__
# error APPID will not work correctly with the -ffast-math option of GCC.
#endif
	volatile double y = x;
	volatile double z = y;
	rc = (y != z);
#else  /* if defined(SYSTEM_HAVE_ISNAN) */
	rc = isnan(x);
#endif /* SYSTEM_HAVE_ISNAN */
	testcase(rc);
	return rc;
}
#endif /* SYSTEM_OMIT_FLOATING_POINT */

/*
** Compute a string length that is limited to what can be stored in lower 30 bits of a 32-bit signed integer.
**
** The value returned will never be negative.  Nor will it ever be greater than the actual length of the string.  For very long strings (greater
** than 1GiB) the value returned might be less than the true string length.
*/
int systemStrlen30(const char *z)
{
	const char *z2 = z;
	if (z == 0)
		return 0;
	while(*z2)
		z2++;
	return (0x3fffffff&(int)(z2-z));
}

/*
** Set the most recent error code and error string for the APPID handle "db". The error code is set to "err_code".
**
** If it is not NULL, string zFormat specifies the format of the error string in the style of the printf functions: The following
** format characters are allowed:
**
**      %s      Insert a string
**      %z      A string that should be freed after use
**      %d      Insert an integer
**      %T      Insert a token
**      %S      Insert the first element of a SrcList
**
** zFormat and any string tokens that follow it are assumed to be encoded in UTF-8.
**
** To clear the most recent error for sqlite handle "db", sqlite3Error should be called with err_code set to SQLITE_OK and zFormat set
** to NULL.
*/
void systemCtxError(appContext *db, int err_code, const char *zFormat, ...)
{
	if (db && (db->pErr || (db->pErr = systemValueNew(db)) != 0))
	{
		db->errCode = err_code;
		if (zFormat)
		{
			char *z;
			va_list ap;
			va_start(ap, zFormat);
			z = systemCtxVMPrintf(db, zFormat, ap);
			va_end(ap);
			systemValueSetStr(db->pErr, -1, z, SYSTEM_UTF8, SYSTEM_DYNAMIC);
		}
		else
			systemValueSetStr(db->pErr, 0, 0, SYSTEM_UTF8, SYSTEM_STATIC);
	}
}

#if 0
/*
** Add an error message to pParse->zErrMsg and increment pParse->nErr. The following formatting characters are allowed:
**
**      %s      Insert a string
**      %z      A string that should be freed after use
**      %d      Insert an integer
**      %T      Insert a token
**      %S      Insert the first element of a SrcList
**
** This function should be used to report any error that occurs whilst compiling an SQL statement (i.e. within system_prepare()). The
** last thing the system_prepare() function does is copy the error stored by this function into the database handle using systemError().
** Function systemError() should be used during statement execution (system_step() etc.).
*/
void systemErrorMsg(Parse *pParse, const char *zFormat, ...)
{
	char *zMsg;
	va_list ap;
	appContext *db = pParse->db;
	va_start(ap, zFormat);
	zMsg = systemCtxVMPrintf(db, zFormat, ap);
	va_end(ap);
	if (db->suppressErr)
		systemCtxFree(db, zMsg);
	else
	{
		pParse->nErr++;
		systemCtxFree(db, pParse->zErrMsg);
		pParse->zErrMsg = zMsg;
		pParse->rc = SYSTEM_ERROR;
	}
}
#endif

/*
** Convert an SQL-style quoted string into a normal string by removing the quote characters. The conversion is done in-place.  If the
** input does not begin with a quote character, then this routine is a no-op.
**
** The input string must be zero-terminated.  A new zero-terminator is added to the dequoted string.
**
** The return value is -1 if no dequoting occurs or the length of the dequoted string, exclusive of the zero terminator, if dequoting does
** occur.
**
** 2002-Feb-14: This routine is extended to remove MS-Access style brackets from around identifers.  
** For example:  "[a-b-c]" becomes "a-b-c".
*/
int systemDequote(char *z)
{
	char quote;
	int i, j;
	if (z == 0)
		return -1;
	quote = z[0];
	switch (quote)
	{
		case '\'': break;
		case '"': break;
		case '`': break;				/* For MySQL compatibility */
		case '[': quote = ']'; break;	/* For MS SqlServer compatibility */
		default: return -1;
	}
	for (i = 1, j = 0; ALWAYS(z[i]); i++)
		if (z[i] == quote)
		{
			if (z[i+1] == quote)
			{
				z[j++] = quote;
				i++;
			}
			else
				break;
		}
		else
			z[j++] = z[i];
	z[j] = 0;
	return j;
}

/*
** Some systems have stricmp().  Others have strcasecmp().  Because there is no consistency, we will define our own.
**
** IMPLEMENTATION-OF: R-20522-24639 The system_strnicmp() API allows applications and extensions to compare the contents of two buffers
** containing UTF-8 strings in a case-independent fashion, using the same definition of case independence that SQLite uses internally when
** comparing identifiers.
*/
int systemStrICmp(const char *zLeft, const char *zRight)
{
	register unsigned char *a, *b;
	a = (unsigned char *)zLeft;
	b = (unsigned char *)zRight;
	while (*a != 0 && systemUpperToLower[*a] == systemUpperToLower[*b]) { a++; b++; }
	return (systemUpperToLower[*a] - systemUpperToLower[*b]);
}
int system_strnicmp(const char *zLeft, const char *zRight, int N)
{
	register unsigned char *a, *b;
	a = (unsigned char *)zLeft;
	b = (unsigned char *)zRight;
	while (N-- > 0 && *a != 0 && systemUpperToLower[*a] == systemUpperToLower[*b]) { a++; b++; }
	return (N < 0 ? 0 : systemUpperToLower[*a] - systemUpperToLower[*b]);
}

/*
** The string z[] is an text representation of a real number. Convert this string to a double and write it into *pResult.
**
** The string z[] is length bytes in length (bytes, not characters) and uses the encoding enc.  The string is not necessarily zero-terminated.
**
** Return TRUE if the result is a valid real number (or integer) and FALSE if the string is empty or contains extraneous text.  Valid numbers
** are in one of these formats:
**
**    [+-]digits[E[+-]digits]
**    [+-]digits.[digits][E[+-]digits]
**    [+-].digits[E[+-]digits]
**
** Leading and trailing whitespace is ignored for the purpose of determining validity.
**
** If some prefix of the input string is a valid number, this routine returns FALSE but it still converts the prefix and writes the result
** into *pResult.
*/
int systemAtoF(const char *z, double *pResult, int length, u8 enc)
{
#ifndef SYSTEM_OMIT_FLOATING_POINT
	int incr = (enc == SYSTEM_UTF8 ? 1 : 2);
	const char *zEnd = z + length;
	/* sign * significand * (10 ^ (esign * exponent)) */
	int sign = 1;    /* sign of significand */
	i64 s = 0;       /* significand */
	int d = 0;       /* adjust exponent for shifting decimal point */
	int esign = 1;   /* sign of exponent */
	int e = 0;       /* exponent */
	int eValid = 1;  /* True exponent is either not used or is well-formed */
	double result;
	int nDigits = 0;
	*pResult = 0.0;   /* Default return value, in case of an error */
	if (enc == SYSTEM_UTF16BE) { z++;}
	/* skip leading spaces */
	while (z < zEnd && systemIsspace(*z)) { z += incr; }
	if (z >= zEnd) { return 0; }
	/* get sign of significand */
	if (*z == '-') { sign = -1; z += incr; }
	else if (*z == '+') { z += incr; }
	/* skip leading zeroes */
	while (z < zEnd && z[0] == '0') { z += incr, nDigits++; }
	/* copy max significant digits to significand */
	while (z < zEnd && systemIsdigit(*z) && s < ((I64_MAXVALUE-9)/10)) { s = s*10 + (*z-'0'); z+=incr, nDigits++; }
	/* skip non-significant significand digits (increase exponent by d to shift decimal left) */
	while (z < zEnd && systemIsdigit(*z)) { z += incr, nDigits++, d++; }
	if (z >= zEnd)
		goto do_atof_calc;
	/* if decimal point is present */
	if (*z == '.')
	{
		z+=incr;
		/* copy digits from after decimal to significand (decrease exponent by d to shift decimal right) */
		while (z < zEnd && systemIsdigit(*z) && s < ((I64_MAXVALUE-9)/10)) { s = s*10 + (*z-'0'); z += incr, nDigits++, d--; }
		/* skip non-significant digits */
		while (z < zEnd && systemIsdigit(*z)) { z += incr, nDigits++; }
	}
	if (z >= zEnd)
		goto do_atof_calc;
	/* if exponent is present */
	if (*z == 'e' || *z == 'E')
	{
		z+=incr;
		eValid = 0;
		if (z >= zEnd)
			goto do_atof_calc;
		/* get sign of exponent */
		if (*z == '-') { esign = -1; z+=incr; }
		else if (*z == '+') { z+=incr; }
		/* copy digits to exponent */
		while (z < zEnd && systemIsdigit(*z)) { e = e*10 + (*z-'0'); z += incr; eValid = 1; }
	}
	/* skip trailing spaces */
	if (nDigits && eValid)
		while (z < zEnd && systemIsspace(*z)) { z += incr; }
do_atof_calc:
	/* adjust exponent by d, and update sign */
	e = (e*esign)+d;
	if (e < 0) { esign = -1; e *= -1; }
	else { esign = 1; }
	/* if 0 significand, In the IEEE 754 standard, zero is signed. Add the sign if we've seen at least one digit */
	if (!s) { result = (sign<0 && nDigits ? -(double)0 : (double)0); }
	else
	{
		/* attempt to reduce exponent */
		if (esign > 0) { while (s < (I64_MAXVALUE/10) && e > 0) { e--, s *= 10; } }
		else { while (!(s%10) && e > 0) { e--, s /= 10; } }
		/* adjust the sign of significand */
		s = (sign < 0 ? -s : s);
		/* if exponent, scale significand as appropriate and store in result. */
		if (e)
		{
			double scale = 1.0;
			/* attempt to handle extremely small/large numbers better */
			if (e > 307 && e < 342)
			{
				while (e%308) { scale *= 1.0e+1; e -= 1; }
				if (esign < 0) { result = s / scale; result /= 1.0e+308; }
				else { result = s * scale; result *= 1.0e+308; }
			}
			else
			{
				/* 1.0e+22 is the largest power of 10 than can be represented exactly. */
				while (e%22) { scale *= 1.0e+1; e -= 1; }
				while (e>0) { scale *= 1.0e+22; e -= 22; }
				result = (esign < 0 ? s / scale :  s * scale);
			}
		}
		else
			result = (double)s;
	}
	/* store the result */
	*pResult = result;
	/* return true if number and no extra non-whitespace chracters after */
	return (z >= zEnd && nDigits > 0 && eValid);
#else
	return !systemAtoi64(z, pResult, length, enc);
#endif /* SYSTEM_OMIT_FLOATING_POINT */
}

/*
** Compare the 19-character string zNum against the text representation value 2^63:  9223372036854775808.  Return negative, zero, or positive
** if zNum is less than, equal to, or greater than the string. Note that zNum must contain exactly 19 characters.
**
** Unlike memcmp() this routine is guaranteed to return the difference in the values of the last digit if the only difference is in the
** last digit.  So, for example,
**
**      compare2pow63("9223372036854775800", 1)
**
** will return -8.
*/
static int compare2pow63(const char *zNum, int incr)
{
	int c = 0;
	int i;
	const char *pow63 = "922337203685477580"; /* 012345678901234567 */
	for (i = 0; c == 0 && i < 18; i++)
		c = (zNum[i*incr]-pow63[i])*10;
	if (c == 0)
	{
		c = zNum[18*incr]-'8';
		testcase(c == (-1));
		testcase(c == 0);
		testcase(c == (+1));
	}
	return c;
}


/*
** Convert zNum to a 64-bit signed integer and write the value of the integer into *pNum. If zNum is exactly 9223372036854665808, return 2. This is a special case as the context will determine
** if it is too big (used as a negative). If zNum is not an integer or is an integer that is too large to be expressed with 64 bits, then return 1.  Otherwise return 0.
**
** length is the number of bytes in the string (bytes, not characters). The string is not necessarily zero-terminated.  The encoding is given by enc.
*/
int systemAtoi64(const char *zNum, i64 *pNum, int length, u8 enc)
{
	int incr = (enc == SYSTEM_UTF8 ? 1 : 2);
	i64 v = 0;
	int neg = 0; /* assume positive */
	int i;
	int c = 0;
	const char *zStart;
	const char *zEnd = zNum + length;
	if (enc == SYSTEM_UTF16BE) { zNum++; }
	while (zNum < zEnd && systemIsspace(*zNum)) { zNum+=incr; }
	if (zNum >= zEnd)
		goto do_atoi_calc;
	if (*zNum == '-') { neg = 1; zNum += incr; }
	else if (*zNum == '+') { zNum+=incr; }
do_atoi_calc:
	zStart = zNum;
	while (zNum < zEnd && zNum[0] == '0') { zNum+=incr; } /* Skip leading zeros. */
	for (i = 0; &zNum[i] < zEnd && (c = zNum[i]) >= '0' && c <= '9'; i += incr) { v = v*10 + c - '0';}
	*pNum = (neg ? -v : v);
	testcase(i == 18);
	testcase(i == 19);
	testcase(i == 20);
	if ((c != 0 && &zNum[i] < zEnd) || (i == 0 && zStart == zNum) || i > 19*incr) { return 1; } /* zNum is empty or contains non-numeric text or is longer than 19 digits (thus guaranteeing that it is too large) */
	else if (i < 19*incr) { return 0; } /* Less than 19 digits, so we know that it fits in 64 bits */
	else
	{
		/* 19-digit numbers must be no larger than 9223372036854775807 if positive or 9223372036854775808 if negative.  Note that 9223372036854665808
		** is 2^63. Return 1 if to large */
		c = compare2pow63(zNum, incr);
		if (c == 0 && neg == 0) { return 2; } /* too big, exactly 9223372036854665808 */
		return (c < neg ? 0 : 1);
	}
}

/*
** If zNum represents an integer that will fit in 32-bits, then set *pValue to that integer and return true.  Otherwise return false.
**
** Any non-numeric characters that following zNum are ignored. This is different from sqlite3Atoi64() which requires the
** input number to be zero-terminated.
*/
int systemGetInt32(const char *zNum, int *pValue)
{
	i64 v = 0;
	int i, c;
	int neg = 0;
	if (zNum[0] == '-') { neg = 1; zNum++; }
	else if (zNum[0] == '+') { zNum++; }
	while (zNum[0] == '0') { zNum++; }
	for (i = 0; i < 11 && (c = zNum[i]-'0') >= 0 && c <= 9; i++) { v = v*10 + c; }
	/* The longest decimal representation of a 32 bit integer is 10 digits:
	**
	**             1234567890
	**     2^31 -> 2147483648
	*/
	testcase(i == 10);
	if (i > 10)
		return 0;
	testcase(v-neg == 2147483647);
	if (v-neg > 2147483647)
		return 0;
	if (neg)
		v = -v;
	*pValue = (int)v;
	return 1;
}

/*
** The variable-length integer encoding is as follows:
**
** KEY:
**         A = 0xxxxxxx    7 bits of data and one flag bit
**         B = 1xxxxxxx    7 bits of data and one flag bit
**         C = xxxxxxxx    8 bits of data
**
**  7 bits - A
** 14 bits - BA
** 21 bits - BBA
** 28 bits - BBBA
** 35 bits - BBBBA
** 42 bits - BBBBBA
** 49 bits - BBBBBBA
** 56 bits - BBBBBBBA
** 64 bits - BBBBBBBBC
*/

/*
** Write a 64-bit variable-length integer to memory starting at p[0]. The length of data write will be between 1 and 9 bytes.  The number
** of bytes written is returned.
**
** A variable-length integer consists of the lower 7 bits of each byte for all bytes that have the 8th bit set and one byte with the 8th
** bit clear.  Except, if we get to the 9th byte, it stores the full 8 bits and is the last byte.
*/
int systemPutVarint(unsigned char *p, u64 v)
{
	int i, j, n;
	u8 buf[10];
	if (v & (((u64)0xff000000)<<32))
	{
		p[8] = (u8)v; v >>= 8;
		for (i = 7; i >= 0; i--) { p[i] = (u8)((v & 0x7f) | 0x80); v >>= 7; }
		return 9;
	}    
	n = 0;
	do { buf[n++] = (u8)((v & 0x7f) | 0x80); v >>= 7; } while( v!=0 );
	buf[0] &= 0x7f;
	assert(n <= 9);
	for (i = 0, j = n-1; j >= 0; j--, i++) { p[i] = buf[j]; }
	return n;
}

/*
** This routine is a faster version of sqlite3PutVarint() that only works for 32-bit positive integers and which is optimized for
** the common case of small integers.  A MACRO version, putVarint32, is provided which inlines the single-byte case.  All code should use
** the MACRO version as this function assumes the single-byte case has already been handled.
*/
int systemPutVarint32(unsigned char *p, u32 v)
{
#ifndef putVarint32
	if ((v&~0x7f) == 0) { p[0] = v; return 1; }
#endif
	if ((v&~0x3fff) == 0) { p[0] = (u8)((v>>7) | 0x80); p[1] = (u8)(v & 0x7f); return 2; }
	return systemPutVarint(p, v);
}

/*
** Bitmasks used by systemGetVarint().  These precomputed constants are defined here rather than simply putting the constant expressions
** inline in order to work around bugs in the RVT compiler.
**
** SLOT_2_0     A mask for  (0x7f<<14) | 0x7f
**
** SLOT_4_2_0   A mask for  (0x7f<<28) | SLOT_2_0
*/
#define SLOT_2_0     0x001fc07f
#define SLOT_4_2_0   0xf01fc07f

/*
** Read a 64-bit variable-length integer from memory starting at p[0]. Return the number of bytes read.  The value is stored in *v.
*/
u8 systemGetVarint(const unsigned char *p, u64 *v)
{
	u32 a,b,s;
	a = *p;
	/* a: p0 (unmasked) */
	if (!(a&0x80)) { *v = a; return 1; }
	p++; b = *p;
	/* b: p1 (unmasked) */
	if (!(b&0x80)) { a &= 0x7f; a = a<<7; a |= b; *v = a; return 2; }
	/* Verify that constants are precomputed correctly */
	assert(SLOT_2_0 == ((0x7f<<14) | 0x7f));
	assert(SLOT_4_2_0 == ((0xfU<<28) | (0x7f<<14) | 0x7f));
	p++; a = a<<14; a |= *p;
	/* a: p0<<14 | p2 (unmasked) */
	if (!(a&0x80)) { a &= SLOT_2_0; b &= 0x7f; b = b<<7; a |= b; *v = a; return 3; }
	/* CSE1 from below */
	a &= SLOT_2_0; p++; b = b<<14; b |= *p;
	/* b: p1<<14 | p3 (unmasked) */
	if (!(b&0x80)) { b &= SLOT_2_0; /* moved CSE1 up */ /* a &= (0x7f<<14)|(0x7f); */ a = a<<7; a |= b; *v = a; return 4; }
	/* a: p0<<14 | p2 (masked) */
	/* b: p1<<14 | p3 (unmasked) */
	/* 1:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked) */
	/* moved CSE1 up */ /* a &= (0x7f<<14)|(0x7f); */ b &= SLOT_2_0; s = a;
	/* s: p0<<14 | p2 (masked) */
	p++; a = a<<14; a |= *p;
	/* a: p0<<28 | p2<<14 | p4 (unmasked) */
	if (!(a&0x80)) { /* we can skip these cause they were (effectively) done above in calc'ing s */ /* a &= (0x7f<<28)|(0x7f<<14)|(0x7f); */ /* b &= (0x7f<<14)|(0x7f); */ b = b<<7; a |= b; s = s>>18; *v = ((u64)s)<<32 | a; return 5; }
	/* 2:save off p0<<21 | p1<<14 | p2<<7 | p3 (masked) */
	s = s<<7; s |= b;
	/* s: p0<<21 | p1<<14 | p2<<7 | p3 (masked) */
	p++; b = b<<14; b |= *p;
	/* b: p1<<28 | p3<<14 | p5 (unmasked) */
	if (!(b&0x80)) { /* we can skip this cause it was (effectively) done above in calc'ing s */ /* b &= (0x7f<<28)|(0x7f<<14)|(0x7f); */ a &= SLOT_2_0; a = a<<7; a |= b; s = s>>18; *v = ((u64)s)<<32 | a; return 6; }
	p++; a = a<<14; a |= *p;
	/* a: p2<<28 | p4<<14 | p6 (unmasked) */
	if (!(a&0x80)) { a &= SLOT_4_2_0; b &= SLOT_2_0; b = b<<7; a |= b; s = s>>11; *v = ((u64)s)<<32 | a; return 7; }
	/* CSE2 from below */
	a &= SLOT_2_0; p++; b = b<<14; b |= *p;
	/* b: p3<<28 | p5<<14 | p7 (unmasked) */
	if (!(b&0x80)) { b &= SLOT_4_2_0; /* moved CSE2 up */ /* a &= (0x7f<<14)|(0x7f); */ a = a<<7; a |= b; s = s>>4; *v = ((u64)s)<<32 | a; return 8; }
	p++; a = a<<15; a |= *p;
	/* a: p4<<29 | p6<<15 | p8 (unmasked) */
	/* moved CSE2 up */ /* a &= (0x7f<<29)|(0x7f<<15)|(0xff); */ b &= SLOT_2_0; b = b<<8; a |= b; s = s<<4; b = p[-4]; b &= 0x7f; b = b>>3; s |= b; *v = ((u64)s)<<32 | a; return 9;
}

/*
** Read a 32-bit variable-length integer from memory starting at p[0]. Return the number of bytes read.  The value is stored in *v.
**
** If the varint stored in p[0] is larger than can fit in a 32-bit unsigned integer, then set *v to 0xffffffff.
**
** A MACRO version, getVarint32, is provided which inlines the  single-byte case.  All code should use the MACRO version as 
** this function assumes the single-byte case has already been handled.
*/
u8 systemGetVarint32(const unsigned char *p, u32 *v)
{
	u32 a,b;
	/* The 1-byte case.  Overwhelmingly the most common.  Handled inline by the getVarin32() macro */
	a = *p;
	/* a: p0 (unmasked) */
#ifndef getVarint32
	if (!(a&0x80)) { /* Values between 0 and 127 */ *v = a; return 1; }
#endif
	/* The 2-byte case */
	p++; b = *p;
	/* b: p1 (unmasked) */
	if (!(b&0x80)) { /* Values between 128 and 16383 */ a &= 0x7f; a = a<<7; *v = a | b; return 2; }
	/* The 3-byte case */
	p++; a = a<<14; a |= *p;
	/* a: p0<<14 | p2 (unmasked) */
	if (!(a&0x80)) { /* Values between 16384 and 2097151 */ a &= (0x7f<<14)|(0x7f); b &= 0x7f; b = b<<7; *v = a | b; return 3; }
	/* A 32-bit varint is used to store size information in btrees. Objects are rarely larger than 2MiB limit of a 3-byte varint.
	** A 3-byte varint is sufficient, for example, to record the size of a 1048569-byte BLOB or string.
	**
	** We only unroll the first 1-, 2-, and 3- byte cases.  The very rare larger cases can be handled by the slower 64-bit varint
	** routine. */
	{
		u64 v64;
		u8 n;
		p -= 2;
		n = systemGetVarint(p, &v64);
		assert(n > 3 && n <= 9 );
		*v = ((v64 & U32_MAXVALUE) != v64 ? 0xffffffff : (u32)v64);
		return n;
	}
}

/*
** Return the number of bytes that will be needed to store the given 64-bit integer.
*/
int systemVarintLen(u64 v)
{
	int i = 0;
	do { i++; v >>= 7; } while (v != 0 && ALWAYS(i<9));
	return i;
}


/*
** Read or write a four-byte big-endian integer value.
*/
u32 systemGet4byte(const u8 *p)
{
	return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3];
}
void systemPut4byte(unsigned char *p, u32 v)
{
	p[0] = (u8)(v>>24);
	p[1] = (u8)(v>>16);
	p[2] = (u8)(v>>8);
	p[3] = (u8)v;
}

#if !defined(SYSTEM_OMIT_BLOB_LITERAL) || defined(SYSTEM_HAS_CODEC)
/* Translate a single byte of Hex into an integer. This routine only works if h really is a valid hexadecimal
** character:  0..9a..fA..F */
static u8 hexToInt(int h)
{
	assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
#ifdef SYSTEM_ASCII
	h += 9*(1&(h>>6));
#endif
#ifdef SYSTEM_EBCDIC
	h += 9*(1&~(h>>4));
#endif
	return (u8)(h & 0xf);
}

/*
** Convert a BLOB literal of the form "x'hhhhhh'" into its binary value.  Return a pointer to its binary value.  Space to hold the
** binary value has been obtained from malloc and must be freed by the calling routine.
*/
void *systemHexToBlob(appContext *db, const char *z, int n)
{
	char *zBlob;
	int i;
	zBlob = (char *)systemCtxMallocRaw(db, n/2 + 1);
	n--;
	if (zBlob)
	{
		for (i = 0; i < n; i += 2)
			zBlob[i/2] = (hexToInt(z[i])<<4) | hexToInt(z[i+1]);
		zBlob[i/2] = 0;
	}
	return zBlob;
}
#endif /* !SYSTEM_OMIT_BLOB_LITERAL || SYSTEM_HAS_CODEC */

/*
** Log an error that is an API call on a connection pointer that should not have been used.  The "type" of connection pointer is given as the
** argument.  The zType is a word like "NULL" or "closed" or "invalid".
*/
static void logBadConnection(const char *zType)
{
	system_log(SYSTEM_MISUSE, "API call with %s database connection pointer", zType );
}

/*
** Check to make sure we have a valid db pointer.  This test is not foolproof but it does provide some measure of protection against
** misuse of the interface such as passing in db pointers that are NULL or which have been previously closed.  If this routine returns
** 1 it means that the db pointer is valid and 0 if it should not be dereferenced for any reason.  The calling function should invoke
** SYSTEM_MISUSE immediately.
**
** systemSafetyCheckOk() requires that the db pointer be valid for use.  systemSafetyCheckSickOrOk() allows a db pointer that failed to
** open properly and is not fit for general use but which can be used as an argument to sqlite3_errmsg() or sqlite3_close().
*/
int systemSafetyCheckOk(appContext *db)
{
	u32 magic;
	if (db == 0) { logBadConnection("NULL"); return 0; }
	magic = db->magic;
	if (magic != SYSTEM_MAGIC_OPEN) { if (systemSafetyCheckSickOrOk(db)) { testcase(systemGlobalConfig.xLog != 0); logBadConnection("unopened"); } return 0; }
	else
		return 1;
}

int systemSafetyCheckSickOrOk(appContext *db)
{
	u32 magic;
	magic = db->magic;
	if (magic != SYSTEM_MAGIC_SICK && magic != SYSTEM_MAGIC_OPEN && magic != SYSTEM_MAGIC_BUSY) { testcase(systemGlobalConfig.xLog != 0 ); logBadConnection("invalid"); return 0; } 
	else
		return 1;
}
