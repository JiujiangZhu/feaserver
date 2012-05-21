#ifndef _SYSTEMAPI_RESET_H_
#define _SYSTEMAPI_RESET_H_
#include <stdarg.h>     /* Needed for the definition of va_list */
#ifdef __cplusplus
extern "C" {
#endif

/*
** Add the ability to override 'extern'
*/
#ifndef SYSTEM_EXTERN
# define SYSTEM_EXTERN extern
#endif

#ifndef SYSTEM_API
# define SYSTEM_API
#endif

/*
** These no-op macros are used in front of interfaces to mark those interfaces as either deprecated or experimental.  New applications
** should not use deprecated interfaces - they are support for backwards compatibility only.  Application writers should be aware that
** experimental interfaces are subject to change in point releases.
**
** These macros used to resolve to various kinds of compiler magic that would generate warning messages when they were used.  But that
** compiler magic ended up generating such a flurry of bug reports that we have taken it all out and gone back to using simple
** noop macros.
*/
#define SYSTEM_DEPRECATED
#define SYSTEM_EXPERIMENTAL

/*
** API: Compile-Time Library Version Numbers
**
** ^(The [SYSTEM_VERSION] C preprocessor macro in this header evaluates to a string literal that is the APPID version in the
** format "X.Y.Z" where X is the major version number and Y is the minor version number and Z is the release number.)^
** ^(The [SYSTEM_VERSIONID] C preprocessor macro resolves to an integer with the value (X*1000000 + Y*1000 + Z) where X, Y, and Z are the same
** numbers used in [SYSTEM_VERSION].)^
*/
#ifdef SYSTEM_VERSION
# undef SYSTEM_VERSION
#endif
#ifdef SYSTEM_VERSIONID
# undef SYSTEM_VERSIONID
#endif
#define SYSTEM_VERSION        "0.1.1"
#define SYSTEM_VERSIONID      0001001

/*
** API: Initialize The APPID Library
**
** ^The system_initialize() routine initializes the APPID library.  ^The system_shutdown() routine
** deallocates any resources that were allocated by system_initialize(). These routines are designed to aid in process initialization and
** shutdown on embedded systems.  Workstation applications using APPID normally do not need to invoke either of these routines.
**
** A call to system_initialize() is an "effective" call if it is the first time system_initialize() is invoked during the lifetime of
** the process, or if it is the first time system_initialize() is invoked following a call to system_shutdown().  ^(Only an effective call
** of system_initialize() does any initialization.  All other calls are harmless no-ops.)^
**
** A call to system_shutdown() is an "effective" call if it is the first call to system_shutdown() since the last system_initialize().  ^(Only
** an effective call to system_shutdown() does any deinitialization. All other valid calls to system_shutdown() are harmless no-ops.)^
**
** The system_initialize() interface is threadsafe, but system_shutdown() is not.  The system_shutdown() interface must only be called from a
** single thread.  All open [AppContexts] must be closed and all other APPID resources must be deallocated prior to invoking
** system_shutdown().
**
** Among other things, ^system_initialize() will invoke system_os_init().  Similarly, ^system_shutdown()
** will invoke system_os_end().
**
** ^The system_initialize() routine returns [SYSTEM_OK] on success. ^If for some reason, system_initialize() is unable to initialize
** the library (perhaps it is unable to allocate a needed resource such as a mutex) it returns an [error code] other than [SYSTEM_OK].
**
** ^The system_initialize() routine is called internally by many other APPID interfaces so that an application usually does not need to
** invoke system_initialize() directly.  For example, [system_open()] calls system_initialize() so the APPID library will be automatically
** initialized when [system_open()] is called if it has not be initialized already.  ^However, if APPID is compiled with the [SYSTEM_OMIT_AUTOINIT]
** compile-time option, then the automatic calls to system_initialize() are omitted and the application must call system_initialize() directly
** prior to using any other APPID interface.  For maximum portability, it is recommended that applications always invoke system_initialize()
** directly prior to using any other APPID interface.  Future releases of APPID may require this.  In other words, the behavior exhibited
** when APPID is compiled with [SYSTEM_OMIT_AUTOINIT] might become the default behavior in some future release of APPID.
**
** The system_os_init() routine does operating-system specific initialization of the APPID library.  The system_os_end()
** routine undoes the effect of system_os_init().  Typical tasks performed by these routines include allocation or deallocation
** of static resources, initialization of global variables, setting up a default [system_vfs] module, or setting up
** a default configuration using [system_config()].
**
** The application should never invoke either system_os_init() or system_os_end() directly.  The application should only invoke
** system_initialize() and system_shutdown().  The system_os_init() interface is called automatically by system_initialize() and
** system_os_end() is called by system_shutdown().  Appropriate implementations for system_os_init() and system_os_end()
** are built into APPID when it is compiled for Unix, Windows, or OS/2. When [custom builds | built for other platforms]
** (using the [SYSTEM_OS_OTHER=1] compile-time option) the application must supply a suitable implementation for
** system_os_init() and system_os_end().  An application-supplied implementation of system_os_init() or system_os_end()
** must return [SYSTEM_OK] on success and some other [error code] upon failure.
*/
SYSTEM_API int system_initialize(void);
SYSTEM_API int system_shutdown(void);
SYSTEM_API int system_os_init(void);
SYSTEM_API int system_os_end(void);

/*
** API: Run-Time Library Version Numbers
** KEYWORDS: system_version
**
** These interfaces provide the same information as the [SYSTEM_VERSION], and [SYSTEM_VERSIONID] C preprocessor macros but are associated with the library instead of the header file. 
** ^(Cautious programmers might include assert() statements in their application to verify that values returned by these interfaces match the macros in the header,
** and thus insure that the application is compiled with matching library and header files.
**
** <blockquote><pre>
** assert(system_gVersionID()==SYSTEM_VERSIONID);
** assert(strcmp(system_gVersion(), SYSTEM_VERSION)==0);
** </pre></blockquote>)^
**
** ^The system_version[] string constant contains the text of [SYSTEM_VERSION] macro.  ^The system_gVersion() function returns a pointer to the
** to the system_version[] string constant.  The system_gVersion() function is provided for use in DLLs since DLL users usually do not have
** direct access to string constants within the DLL.  ^The system_gVersionID() function returns an integer equal to [SYSTEM_VERSION_NUMBER].
**
** See also: [system_gVersion()].
*/
SYSTEM_API SYSTEM_EXTERN const char system_version[];
SYSTEM_API const char *system_gVersion(void);
SYSTEM_API int system_gVersionID(void);

/*
** API: Run-Time Library Compilation Options Diagnostics
**
** ^The system_GetHasCompileOption() function returns 0 or 1  indicating whether the specified option was defined at 
** compile time.  ^The SYSTEM_ prefix may be omitted from the option name passed to system_GetHasCompileOption().  
**
** ^The system_GetCompileOptionByID() function allows iterating over the list of options that were defined at compile time by
** returning the N-th compile time option string.  ^If N is out of range, system_GetCompileOptionByID() returns a NULL pointer.  ^The SYSTEM_ 
** prefix is omitted from any strings returned by  system_compileoption_get().
**
** ^Support for the diagnostic functions system_compileoption_used() and system_compileoption_get() may be omitted by specifying the 
** [SYSTEM_OMIT_COMPILEOPTION_DIAGS] option at compile time.
**
** See also: SQL functions [sqlite_compileoption_used()] and [sqlite_compileoption_get()] and the [compile_options pragma].
*/
#ifndef SYSTEM_OMIT_COMPILEOPTION_DIAGS
SYSTEM_API int system_GetHasCompileOption(const char *zOptName);
SYSTEM_API const char *system_GetCompileOptionByIndex(int index);
#endif

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_RESET_H_ */

