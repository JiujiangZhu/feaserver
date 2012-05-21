#ifndef __SYSTEMAPI_PRIMITIVES_H_
#define __SYSTEMAPI_PRIMITIVES_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: 64-Bit Integer Types
** KEYWORDS: INT64_TYPE UINT64_TYPE
**
** Because there is no cross-platform way to specify 64-bit integer types APPID includes typedefs for 64-bit signed and unsigned integers.
**
** ^The INT64_TYPE types can store integer values between -9223372036854775808 and +9223372036854775807 inclusive.  ^The
** UINT64_TYPE types can store integer values  between 0 and +18446744073709551615 inclusive.
*/
#if defined(_MSC_VER) || defined(__BORLANDC__)
	typedef __int64 INT64_TYPE;
	typedef unsigned __int64 UINT64_TYPE;
#else
	typedef long long int INT64_TYPE;
	typedef unsigned long long int UINT64_TYPE;
#endif

// If compiling for a processor that lacks floating point support, substitute integer for floating-point.
#ifdef SYSTEM_OMIT_FLOATING_POINT
# define double INT64_TYPE
#endif

/*
** API: Dynamically Typed Value Object
** KEYWORDS: {protected system_value} {unprotected system_value}
**
** APPID uses the system_value object to represent all values that can be stored in a database table. APPID uses dynamic typing
** for the values it stores.  ^Values stored in system_value objects can be integers, floating point values, strings, BLOBs, or NULL.
**
** An system_value object may be either "protected" or "unprotected". Some interfaces require a protected system_value.  Other interfaces
** will accept either a protected or an unprotected system_value. Every interface that accepts system_value arguments specifies
** whether or not it requires a protected system_value.
**
** The terms "protected" and "unprotected" refer to whether or not a mutex is held.  A internal mutex is held for a protected
** system_value object but no mutex is held for an unprotected system_value object.  If APPID is compiled to be single-threaded
** (with [SYSTEM_THREADSAFE=0] and with [system_threadsafe()] returning 0) or if APPID is run in one of reduced mutex modes 
** [SYSTEM_CONFIG_SINGLETHREAD] or [SYSTEM_CONFIG_MULTITHREAD] then there is no distinction between protected and unprotected
** system_value objects and they can be used interchangeably.  However, for maximum code portability it is recommended that applications
** still make the distinction between protected and unprotected system_value objects even when not strictly required.
**
*/
typedef struct Mem system_value;

/*
** API: Formatted String Printing Functions
**
** These routines are work-alikes of the "printf()" family of functions from the standard C library.
**
** ^The system_mprintf() and system_vmprintf() routines write their results into memory obtained from [system_malloc()].
** The strings returned by these two routines should be released by [system_free()].  ^Both routines return a
** NULL pointer if [system_malloc()] is unable to allocate enough memory to hold the resulting string.
**
** ^(In system_snprintf() routine is similar to "snprintf()" from the standard C library.  The result is written into the
** buffer supplied as the second parameter whose size is given by the first parameter. Note that the order of the
** first two parameters is reversed from snprintf().)^  This is an historical accident that cannot be fixed without breaking
** backwards compatibility.  ^(Note also that system_snprintf() returns a pointer to its buffer instead of the number of
** characters actually written into the buffer.)^  We admit that the number of characters written would be a more useful return
** value but we cannot change the implementation of system_snprintf() now without breaking compatibility.
**
** ^As long as the buffer size is greater than zero, system_snprintf() guarantees that the buffer is always zero-terminated.  ^The first
** parameter "n" is the total size of the buffer, including space for the zero terminator.  So the longest string that can be completely
** written will be n-1 characters.
**
** These routines all implement some additional formatting options that are useful for constructing SQL statements.
** All of the usual printf() formatting options apply.  In addition, there is are "%q", "%Q", and "%z" options.
**
** ^(The %q option works like %s in that it substitutes a null-terminated string from the argument list.  But %q also doubles every '\'' character.
** %q is designed for use inside a string literal.)^  By doubling each '\'' character it escapes that character and allows it to be inserted into
** the string.
**
** For example, assume the string variable zText contains text as follows:
**
** <blockquote><pre>
**  char *zText = "It's a happy day!";
** </pre></blockquote>
**
** One can use this text in an SQL statement as follows:
**
** <blockquote><pre>
**  char *zSQL = system_mprintf("INSERT INTO table VALUES('%q')", zText);
**  system_exec(db, zSQL, 0, 0, 0);
**  system_free(zSQL);
** </pre></blockquote>
**
** Because the %q format string is used, the '\'' character in zText is escaped and the SQL generated is as follows:
**
** <blockquote><pre>
**  INSERT INTO table1 VALUES('It''s a happy day!')
** </pre></blockquote>
**
** This is correct.  Had we used %s instead of %q, the generated SQL would have looked like this:
**
** <blockquote><pre>
**  INSERT INTO table1 VALUES('It's a happy day!');
** </pre></blockquote>
**
** This second example is an SQL syntax error.  As a general rule you should always use %q instead of %s when inserting text into a string literal.
**
** ^(The %Q option works like %q except it also adds single quotes around the outside of the total string.  Additionally, if the parameter in the
** argument list is a NULL pointer, %Q substitutes the text "NULL" (without single quotes).)^  So, for example, one could say:
**
** <blockquote><pre>
**  char *zSQL = system_mprintf("INSERT INTO table VALUES(%Q)", zText);
**  system_exec(db, zSQL, 0, 0, 0);
**  system_free(zSQL);
** </pre></blockquote>
**
** The code above will render a correct SQL statement in the zSQL variable even if the zText variable is a NULL pointer.
**
** ^(The "%z" formatting option works like "%s" but with the addition that after the string has been read and copied into
** the result, [system_free()] is called on the input string.)^
*/
SYSTEM_API char *system_mprintf(const char*,...);
SYSTEM_API char *system_vmprintf(const char*, va_list);
SYSTEM_API char *system_snprintf(int,char*,const char*, ...);

/*
** API: Pseudo-Random Number Generator
**
** APPID contains a high-quality pseudo-random number generator (PRNG).
**
** ^A call to this routine stores N bytes of randomness into buffer P.
**
** ^The first time this routine is invoked (either internally or by the application) the PRNG is seeded using randomness obtained
** from the xRandomness method of the default [system_vfs] object. ^On all subsequent invocations, the pseudo-randomness is generated
** internally and without recourse to the [system_vfs] xRandomness method.
*/
SYSTEM_API void system_randomness(int N, void *P);

/*
** API: Fundamental Datatypes
** KEYWORDS: SYSTEM_TEXT
**
** ^(Every value in APPID has one of five fundamental datatypes:
**
** <ul>
** <li> 64-bit signed integer
** <li> 64-bit IEEE floating point number
** <li> string
** <li> BLOB
** <li> NULL
** </ul>)^
**
** These constants are codes for each of those types.
*/
#define SYSTEM_INTEGER  1
#define SYSTEM_FLOAT    2
#define SYSTEM_BLOB     4
#define SYSTEM_NULL     5
#define SYSTEM_TEXT     3

/*
** API: Text Encodings
**
** These constant define integer codes that represent the various text encodings supported by APPID.
*/
#define SYSTEM_UTF8           1
#define SYSTEM_UTF16LE        2
#define SYSTEM_UTF16BE        3
#define SYSTEM_UTF16          4    /* Use native byte order */
#define SYSTEM_ANY            5    /* system_create_function only */
#define SYSTEM_UTF16_ALIGNED  8    /* system_create_collation only */

/*
** API: String Comparison
**
** ^The [system_strnicmp()] API allows applications and extensions to compare the contents of two buffers containing UTF-8 strings in a
** case-independent fashion, using the same definition of case independence that APPID uses internally when comparing identifiers.
*/
SYSTEM_API int system_strnicmp(const char*, const char*, int);


/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Define New Collating Sequences
**
** ^These functions add, remove, or modify a [collation] associated with the [appContext] specified as the first argument.
**
** ^The name of the collation is a UTF-8 string for system_create_collation() and system_create_collation_v2()
** and a UTF-16 string in native byte order for system_create_collation16(). ^Collation names that compare equal according to [system_strnicmp()] are
** considered to be the same name.
**
** ^(The third argument (eTextRep) must be one of the constants:
** <ul>
** <li> [SYSTEM_UTF8],
** <li> [SYSTEM_UTF16LE],
** <li> [SYSTEM_UTF16BE],
** <li> [SYSTEM_UTF16], or
** <li> [SYSTEM_UTF16_ALIGNED].
** </ul>)^
** ^The eTextRep argument determines the encoding of strings passed to the collating function callback, xCallback.
** ^The [SYSTEM_UTF16] and [SYSTEM_UTF16_ALIGNED] values for eTextRep force strings to be UTF16 with native byte order.
** ^The [SYSTEM_UTF16_ALIGNED] value for eTextRep forces strings to begin on an even byte address.
**
** ^The fourth argument, pArg, is a application data pointer that is passed through as the first argument to the collating function callback.
**
** ^The fifth argument, xCallback, is a pointer to the collating function. ^Multiple collating functions can be registered using the same name but
** with different eTextRep parameters and APPID will use whichever function requires the least amount of data transformation.
** ^If the xCallback argument is NULL then the collating function is deleted.  ^When all collating functions having the same name are deleted,
** that collation is no longer usable.
**
** ^The collating function callback is invoked with a copy of the pArg application data pointer and with two strings in the encoding specified
** by the eTextRep argument.  The collating function must return an integer that is negative, zero, or positive
** if the first string is less than, equal to, or greater than the second, respectively.  A collating function must alway return the same answer
** given the same inputs.  If two or more collating functions are registered to the same collation name (using different eTextRep values) then all
** must give an equivalent answer when invoked with equivalent strings. The collating function must obey the following properties for all strings A, B, and C:
**
** <ol>
** <li> If A==B then B==A.
** <li> If A==B and B==C then A==C.
** <li> If A&lt;B THEN B&gt;A.
** <li> If A&lt;B and B&lt;C then A&lt;C.
** </ol>
**
** If a collating function fails any of the above constraints and that collating function is registered and used, then the behavior of APPID is undefined.
**
** ^The system_create_collation_v2() works like system_create_collation() with the addition that the xDestroy callback is invoked on pArg when
** the collating function is deleted. ^Collating functions are deleted when they are overridden by later calls to the collation creation functions or when the
** [database connection] is closed using [system_close()].
**
** See also:  [system_collation_needed()] and [system_collation_needed16()].
*/
SYSTEM_API int system_create_collation(appContext*, const char *zName,  int eTextRep, void *pArg, int(*xCompare)(void*,int,const void*,int,const void*));
SYSTEM_API int system_create_collation16(appContext*, const void *zName, int eTextRep,  void *pArg, int(*xCompare)(void*,int,const void*,int,const void*));
SYSTEM_API int system_create_collation_v2(appContext*, const char *zName,  int eTextRep, void *pArg, int(*xCompare)(void*,int,const void*,int,const void*), void(*xDestroy)(void*));

/*
** API: Collation Needed Callbacks
**
** ^To avoid having to register all collation sequences before a database can be used, a single callback function may be registered with the
** [database connection] to be invoked whenever an undefined collation sequence is required.
**
** ^If the function is registered using the system_collation_needed() API, then it is passed the names of undefined collation sequences as strings
** encoded in UTF-8. ^If system_collation_needed16() is used, the names are passed as UTF-16 in machine native byte order.
** ^A call to either function replaces the existing collation-needed callback.
**
** ^(When the callback is invoked, the first argument passed is a copy of the second argument to system_collation_needed() or
** system_collation_needed16().  The second argument is the database connection.  The third argument is one of [SYSTEM_UTF8], [SYSTEM_UTF16BE],
** or [SYSTEM_UTF16LE], indicating the most desirable form of the collation sequence function required.  The fourth parameter is the name of the
** required collation sequence.)^
**
** The callback function should register the desired collation using [system_create_collation()], [system_create_collation16()], or [system_create_collation_v2()].
*/
SYSTEM_API int system_collation_needed(appContext*, void*, void(*)(void*,appContext*,int eTextRep,const char*));
SYSTEM_API int system_collation_needed16(appContext*, void*, void(*)(void*,appContext*,int eTextRep,const void*));

/*
** API: A Handle To An Open BLOB
** KEYWORDS: {BLOB handle} {BLOB handles}
**
** An instance of this object represents an open BLOB on which [system_blob_open | incremental BLOB I/O] can be performed.
** ^Objects of this type are created by [system_blob_open()] and destroyed by [system_blob_close()].
** ^The [system_blob_read()] and [system_blob_write()] interfaces can be used to read or write small subsections of the BLOB.
** ^The [system_blob_bytes()] interface returns the size of the BLOB in bytes.
*/
typedef struct system_blob system_blob;

/*
** API: Open A BLOB For Incremental I/O
**
** ^(This interfaces opens a [BLOB handle | handle] to the BLOB located in row iRow, column zColumn, table zTable in database zDb;
** in other words, the same BLOB that would be selected by:
**
** <pre>
**     SELECT zColumn FROM zDb.zTable WHERE [rowid] = iRow;
** </pre>)^
**
** ^If the flags parameter is non-zero, then the BLOB is opened for read and write access. ^If it is zero, the BLOB is opened for read access.
** ^It is not possible to open a column that is part of an index or primary key for writing. ^If [foreign key constraints] are enabled, it is 
** not possible to open a column that is part of a [child key] for writing.
**
** ^Note that the database name is not the filename that contains the database but rather the symbolic name of the database that
** appears after the AS keyword when the database is connected using [ATTACH]. ^For the main database file, the database name is "main".
** ^For TEMP tables, the database name is "temp".
**
** ^(On success, [SYSTEM_OK] is returned and the new [BLOB handle] is written to *ppBlob. Otherwise an [error code] is returned and *ppBlob is set
** to be a null pointer.)^ ^This function sets the [app context] error code and message accessible via [system_errcode()] and [system_errmsg()] and related
** functions. ^Note that the *ppBlob variable is always initialized in a way that makes it safe to invoke [system_blob_close()] on *ppBlob
** regardless of the success or failure of this routine.
**
** ^(If the row that a BLOB handle points to is modified by an [UPDATE], [DELETE], or by [ON CONFLICT] side-effects then the BLOB handle is marked as "expired".
** This is true if any column of the row is changed, even a column other than the one the BLOB handle is open on.)^
** ^Calls to [system_blob_read()] and [system_blob_write()] for a expired BLOB handle fail with an return code of [SYSTEM_ABORT].
** ^(Changes written into a BLOB prior to the BLOB expiring are not rolled back by the expiration of the BLOB.  Such changes will eventually
** commit if the transaction continues to completion.)^
**
** ^Use the [system_blob_bytes()] interface to determine the size of the opened blob.  ^The size of a blob may not be changed by this
** interface.  Use the [UPDATE] SQL command to change the size of a blob.
**
** ^The [system_bind_zeroblob()] and [system_result_zeroblob()] interfaces and the built-in [zeroblob] SQL function can be used, if desired,
** to create an empty, zero-filled blob in which to read or write using this interface.
**
** To avoid a resource leak, every open [BLOB handle] should eventually be released by a call to [system_blob_close()].
*/
SYSTEM_API int system_blob_open(appContext*, const char *zDb, const char *zTable, const char *zColumn, INT64_TYPE iRow, int flags, system_blob **ppBlob);

/*
** API: Close A BLOB Handle
**
** ^Closes an open [BLOB handle].
**
** ^Closing a BLOB shall cause the current transaction to commit if there are no other BLOBs, no pending prepared statements, and the
** app context is in [autocommit mode]. ^If any writes were made to the BLOB, they might be held in cache
** until the close operation if they will fit.
**
** ^(Closing the BLOB often forces the changes out to disk and so if any I/O errors occur, they will likely occur
** at the time when the BLOB is closed.  Any errors that occur during closing are reported as a non-zero return value.)^
**
** ^(The BLOB is closed unconditionally.  Even if this routine returns an error code, the BLOB is still closed.)^
**
** ^Calling this routine with a null pointer (such as would be returned by a failed call to [system_blob_open()]) is a harmless no-op.
*/
SYSTEM_API int system_blob_close(system_blob *);

/*
** API: Return The Size Of An Open BLOB
**
** ^Returns the size in bytes of the BLOB accessible via the successfully opened [BLOB handle] in its only argument.  ^The
** incremental blob I/O routines can only read or overwriting existing blob content; they cannot change the size of a blob.
**
** This routine only works on a [BLOB handle] which has been created by a prior successful call to [system_blob_open()] and which has not
** been closed by [system_blob_close()].  Passing any other pointer in to this routine results in undefined and probably undesirable behavior.
*/
SYSTEM_API int system_blob_bytes(system_blob *);

/*
** API: Read Data From A BLOB Incrementally
**
** ^(This function is used to read data from an open [BLOB handle] into a caller-supplied buffer. N bytes of data are copied into buffer Z
** from the open BLOB, starting at offset iOffset.)^
**
** ^If offset iOffset is less than N bytes from the end of the BLOB, [SYSTEM_ERROR] is returned and no data is read.  ^If N or iOffset is
** less than zero, [SYSTEM_ERROR] is returned and no data is read. ^The size of the blob (and hence the maximum value of N+iOffset)
** can be determined using the [system_blob_bytes()] interface.
**
** ^An attempt to read from an expired [BLOB handle] fails with an error code of [SYSTEM_ABORT].
**
** ^(On success, system_blob_read() returns SYSTEM_OK. Otherwise, an [error code] or an [extended error code] is returned.)^
**
** This routine only works on a [BLOB handle] which has been created by a prior successful call to [system_blob_open()] and which has not
** been closed by [system_blob_close()].  Passing any other pointer in to this routine results in undefined and probably undesirable behavior.
**
** See also: [system_blob_write()].
*/
SYSTEM_API int system_blob_read(system_blob *, void *Z, int N, int iOffset);

/*
** API: Write Data Into A BLOB Incrementally
**
** ^This function is used to write data into an open [BLOB handle] from a caller-supplied buffer. ^N bytes of data are copied from the buffer Z
** into the open BLOB, starting at offset iOffset.
**
** ^If the [BLOB handle] passed as the first argument was not opened for writing (the flags parameter to [system_blob_open()] was zero),
** this function returns [SYSTEM_READONLY].
**
** ^This function may only modify the contents of the BLOB; it is not possible to increase the size of a BLOB using this API.
** ^If offset iOffset is less than N bytes from the end of the BLOB, [SYSTEM_ERROR] is returned and no data is written.  ^If N is
** less than zero [SYSTEM_ERROR] is returned and no data is written. The size of the BLOB (and hence the maximum value of N+iOffset)
** can be determined using the [system_blob_bytes()] interface.
**
** ^An attempt to write to an expired [BLOB handle] fails with an error code of [SYSTEM_ABORT].  ^Writes to the BLOB that occurred
** before the [BLOB handle] expired are not rolled back by the expiration of the handle, though of course those changes might
** have been overwritten by the statement that expired the BLOB handle or by other independent statements.
**
** ^(On success, system_blob_write() returns SYSTEM_OK. Otherwise, an  [error code] or an [extended error code] is returned.)^
**
** This routine only works on a [BLOB handle] which has been created by a prior successful call to [system_blob_open()] and which has not
** been closed by [system_blob_close()].  Passing any other pointer in to this routine results in undefined and probably undesirable behavior.
**
** See also: [system_blob_read()].
*/
SYSTEM_API int system_blob_write(system_blob *, const void *z, int n, int iOffset);

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* __SYSTEMAPI_PRIMITIVES_H_ */

