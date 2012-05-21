#ifndef __SYSTEMAPI_APPCONTEXT_H_
#define __SYSTEMAPI_APPCONTEXT_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Database Connection Handle
** KEYWORDS: {app context} {app contexts}
**
** Each open APPID context is represented by a pointer to an instance of the opaque structure named "appContext".  It is useful to think of an appContext
** pointer as an object.  The [system_open()], [system_open16()], and [system_open_v2()] interfaces are its constructors, and [system_close()]
** is its destructor.  There are many other interfaces (such as [system_prepare_v2()], [system_create_function()], and
** [system_busy_timeout()] to name but three) that are methods on an appContext object.
*/
typedef struct appContext appContext;

/*
** API: Opening A New Database Connection
**
** ^These routines open an APPID database file whose name is given by the filename argument. ^The filename argument is interpreted as UTF-8 for
** system_open() and system_open_v2() and as UTF-16 in the native byte order for system_open16(). ^(A [app context] handle is usually
** returned in *ppDb, even if an error occurs.  The only exception is that if APPID is unable to allocate memory to hold the [sqlite3] object,
** a NULL will be written into *ppDb instead of a pointer to the [appContext] object.)^ ^(If the database is opened (and/or created) successfully, then
** [SYSTEM_OK] is returned.  Otherwise an [error code] is returned.)^ ^The [system_errmsg()] or [system_errmsg16()] routines can be used to obtain
** an English language description of the error following a failure of any of the system_open() routines.
**
** ^The default encoding for the database will be UTF-8 if system_open() or system_open_v2() is called and
** UTF-16 in the native byte order if system_open16() is used.
**
** Whether or not an error occurs when it is opened, resources associated with the [app context] handle should be released by
** passing it to [system_close()] when it is no longer required.
**
** The system_open_v2() interface works like system_open() except that it accepts two additional parameters for additional control
** over the new app context.  ^(The flags parameter to system_open_v2() can take one of the following three values, optionally combined with the 
** [SYSTEM_OPEN_NOMUTEX], [SYSTEM_OPEN_FULLMUTEX], [SYSTEM_OPEN_SHAREDCACHE], and/or [SYSTEM_OPEN_PRIVATECACHE] flags:)^
**
** <dl>
** ^(<dt>[SYSTEM_OPEN_READONLY]</dt>
** <dd>The database is opened in read-only mode.  If the database does not already exist, an error is returned.</dd>)^
**
** ^(<dt>[SYSTEM_OPEN_READWRITE]</dt>
** <dd>The database is opened for reading and writing if possible, or reading only if the file is write protected by the operating system.  In either
** case the database must already exist, otherwise an error is returned.</dd>)^
**
** ^(<dt>[SYSTEM_OPEN_READWRITE] | [SYSTEM_OPEN_CREATE]</dt>
** <dd>The database is opened for reading and writing, and is creates it if it does not already exist. This is the behavior that is always used for
** system_open() and system_open16().</dd>)^
** </dl>
**
** If the 3rd parameter to system_open_v2() is not one of the combinations shown above or one of the combinations shown above combined
** with the [SYSTEM_OPEN_NOMUTEX], [SYSTEM_OPEN_FULLMUTEX], [SYSTEM_OPEN_SHAREDCACHE] and/or [SYSTEM_OPEN_PRIVATECACHE] flags,
** then the behavior is undefined.
**
** ^If the [SYSTEM_OPEN_NOMUTEX] flag is set, then the app context opens in the multi-thread [threading mode] as long as the single-thread
** mode has not been set at compile-time or start-time.  ^If the [SYSTEM_OPEN_FULLMUTEX] flag is set then the app context opens
** in the serialized [threading mode] unless single-thread was previously selected at compile-time or start-time.
** ^The [SYSTEM_OPEN_SHAREDCACHE] flag causes the app context to be eligible to use [shared cache mode], regardless of whether or not shared
** cache is enabled using [system_enable_shared_cache()].  ^The [SYSTEM_OPEN_PRIVATECACHE] flag causes the app context to not
** participate in [shared cache mode] even if it is enabled.
**
** ^If the filename is ":memory:", then a private, temporary in-memory database is created for the connection.  ^This in-memory database will vanish when
** the app context is closed.  Future versions of APPID might make use of additional special filenames that begin with the ":" character.
** It is recommended that when a database filename actually does begin with a ":" character you should prefix the filename with a pathname such as
** "./" to avoid ambiguity.
**
** ^If the filename is an empty string, then a private, temporary on-disk database will be created.  ^This private database will be
** automatically deleted as soon as the app context is closed.
**
** ^The fourth parameter to system_open_v2() is the name of the [system_vfs] object that defines the operating system interface that
** the new app context should use.  ^If the fourth parameter is a NULL pointer then the default [system_vfs] object is used.
**
** <b>Note to Windows users:</b>  The encoding used for the filename argument of system_open() and system_open_v2() must be UTF-8, not whatever
** codepage is currently defined.  Filenames containing international characters must be converted to UTF-8 prior to passing them into
** system_open() or system_open_v2().
*/
SYSTEM_API int system_openctx(const char *filename, appContext **ppDb);/* Database filename (UTF-8), OUT: APPID db handle */
SYSTEM_API int system_openctx16(const void *filename, appContext **ppDb);/* Database filename (UTF-16), OUT: APPID db handle */
SYSTEM_API int system_openctx_v2(const char *filename, appContext **ppDb, int flags, const char *zVfs);/* Database filename (UTF-8), OUT: APPID db handle, Flags, Name of VFS module to use */

/*
** API: Closing A Database Connection
**
** ^The system_close() routine is the destructor for the [sqlite3] object.
** ^Calls to system_close() return SYSTEM_OK if the [sqlite3] object is
** successfully destroyed and all associated resources are deallocated.
**
** Applications must [system_finalize | finalize] all [prepared statements]
** and [system_blob_close | close] all [BLOB handles] associated with
** the [sqlite3] object prior to attempting to close the object.  ^If
** system_close() is called on a [app context] that still has
** outstanding [prepared statements] or [BLOB handles], then it returns
** SYSTEM_BUSY.
**
** ^If [system_close()] is invoked while a transaction is open,
** the transaction is automatically rolled back.
**
** The C parameter to [system_close(C)] must be either a NULL
** pointer or an [sqlite3] object pointer obtained
** from [system_open()], [system_open16()], or
** [system_open_v2()], and not previously closed.
** ^Calling system_close() with a NULL pointer argument is a 
** harmless no-op.
*/
SYSTEM_API int system_closectx(appContext *);

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_APPCONTEXT_H_ */


