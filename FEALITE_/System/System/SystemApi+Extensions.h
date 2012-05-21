#ifndef _SYSTEMAPI_EXTENSIONS_H_
#define _SYSTEMAPI_EXTENSIONS_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Automatically Load Statically Linked Extensions
**
** ^This interface causes the xEntryPoint() function to be invoked for each new [app context] that is created.  The idea here is that
** xEntryPoint() is the entry point for a statically linked APPID extension that is to be automatically loaded into all new app contexts.
**
** ^(Even though the function prototype shows that xEntryPoint() takes no arguments and returns void, APPID invokes xEntryPoint() with three
** arguments and expects and integer result as if the signature of the entry point where as follows:
**
** <blockquote><pre>
** &nbsp;  int xEntryPoint(
** &nbsp;    sqlite3 *db,
** &nbsp;    const char **pzErrMsg,
** &nbsp;    const struct system_api_routines *pThunk
** &nbsp;  );
** </pre></blockquote>)^
**
** If the xEntryPoint routine encounters an error, it should make *pzErrMsg point to an appropriate error message (obtained from [system_mprintf()])
** and return an appropriate [error code].  ^APPID ensures that *pzErrMsg is NULL before calling the xEntryPoint().  ^APPID will invoke
** [system_free()] on *pzErrMsg after xEntryPoint() returns.  ^If any xEntryPoint() returns an error, the [system_open()], [system_open16()],
** or [system_open_v2()] call that provoked the xEntryPoint() will fail.
**
** ^Calling system_auto_extension(X) with an entry point X that is already on the list of automatic extensions is a harmless no-op. ^No entry point
** will be called more than once for each app context that is opened.
**
** See also: [system_reset_auto_extension()].
*/
SYSTEM_API int system_auto_extension(void (*xEntryPoint)(void));

/*
** API: Reset Automatic Extension Loading
**
** ^This interface disables all automatic extensions previously registered using [system_auto_extension()].
*/
SYSTEM_API void system_reset_auto_extension(void);


/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Load An Extension
**
** ^This interface loads an APPID extension library from the named file.
**
** ^The system_load_extension() interface attempts to load an APPID extension library contained in the file zFile.
**
** ^The entry point is zProc. ^zProc may be 0, in which case the name of the entry point
** defaults to "system_extension_init". ^The system_load_extension() interface returns
** [SYSTEM_OK] on success and [SYSTEM_ERROR] if something goes wrong. ^If an error occurs and pzErrMsg is not 0, then the
** [system_load_extension()] interface shall attempt to fill *pzErrMsg with error message text stored in memory
** obtained from [system_malloc()]. The calling function should free this memory by calling [system_free()].
**
** ^Extension loading must be enabled using [system_enable_load_extension()] prior to calling this API,
** otherwise an error will be returned.
**
** See also the [load_extension() SQL function].
*/
SYSTEM_API int system_load_extension(
	appContext *ctx,          /* Load the extension into this app context */
	const char *zFile,    /* Name of the shared library containing extension */
	const char *zProc,    /* Entry point.  Derived from zFile if 0 */
	char **pzErrMsg       /* Put error message here if not 0 */
);

/*
** API: Enable Or Disable Extension Loading
**
** ^So as not to open security holes in older applications that are unprepared to deal with extension loading, and as a means of disabling
** extension loading while evaluating user-entered SQL, the following API is provided to turn the [system_load_extension()] mechanism on and off.
**
** ^Extension loading is off by default. See ticket #1863. ^Call the system_enable_load_extension() routine with onoff==1
** to turn extension loading on and call it with onoff==0 to turn it back off again.
*/
SYSTEM_API int system_enable_load_extension(appContext *ctx, int onoff);

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_EXTENSIONS_H_ */
