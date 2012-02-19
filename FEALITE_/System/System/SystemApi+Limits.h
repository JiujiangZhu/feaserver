#ifndef _SYSTEMAPI_LIMITS_H_
#define _SYSTEMAPI_LIMITS_H_
#ifdef __cplusplus
extern "C" {
#endif

/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Compile-Time Authorization Callbacks
**
** ^This routine registers a authorizer callback with a particular [app context], supplied in the first argument.
** ^The authorizer callback is invoked as SQL statements are being compiled by [system_prepare()] or its variants [system_prepare_v2()],
** [system_prepare16()] and [system_prepare16_v2()].  ^At various points during the compilation process, as logic is being created
** to perform various actions, the authorizer callback is invoked to see if those actions are allowed.  ^The authorizer callback should
** return [SYSTEM_OK] to allow the action, [SYSTEM_IGNORE] to disallow the specific action but allow the SQL statement to continue to be
** compiled, or [SYSTEM_DENY] to cause the entire SQL statement to be rejected with an error.  ^If the authorizer callback returns
** any value other than [SYSTEM_IGNORE], [SYSTEM_OK], or [SYSTEM_DENY] then the [system_prepare_v2()] or equivalent call that triggered
** the authorizer will fail with an error message.
**
** When the callback returns [SYSTEM_OK], that means the operation requested is ok.  ^When the callback returns [SYSTEM_DENY], the
** [system_prepare_v2()] or equivalent call that triggered the authorizer will fail with an error message explaining that
** access is denied. 
**
** ^The first parameter to the authorizer callback is a copy of the third parameter to the system_set_authorizer() interface. ^The second parameter
** to the callback is an integer [SYSTEM_COPY | action code] that specifies the particular action to be authorized. ^The third through sixth parameters
** to the callback are zero-terminated strings that contain additional details about the action to be authorized.
**
** ^If the action code is [SYSTEM_READ] and the callback returns [SYSTEM_IGNORE] then the
** [prepared statement] statement is constructed to substitute a NULL value in place of the table column that would have
** been read if [SYSTEM_OK] had been returned.  The [SYSTEM_IGNORE] return can be used to deny an untrusted user access to individual
** columns of a table. ^If the action code is [SYSTEM_DELETE] and the callback returns [SYSTEM_IGNORE] then the [DELETE] operation proceeds but the
** [truncate optimization] is disabled and all rows are deleted individually.
**
** An authorizer is used when [system_prepare | preparing] SQL statements from an untrusted source, to ensure that the SQL statements
** do not try to access data they are not allowed to see, or that they do not try to execute malicious statements that damage the database.  For
** example, an application may allow a user to enter arbitrary SQL queries for evaluation by a database.  But the application does
** not want the user to be able to make arbitrary changes to the database.  An authorizer could then be put in place while the
** user-entered SQL is being [system_prepare | prepared] that disallows everything except [SELECT] statements.
**
** Applications that need to process SQL from untrusted sources might also consider lowering resource limits using [system_limit()]
** and limiting database size using the [max_page_count] [PRAGMA] in addition to using an authorizer.
**
** ^(Only a single authorizer can be in place on a app context at a time.  Each call to system_set_authorizer overrides the
** previous call.)^  ^Disable the authorizer by installing a NULL callback. The authorizer is disabled by default.
**
** The authorizer callback must not do anything that will modify the app context that invoked the authorizer callback.
** Note that [system_prepare_v2()] and [system_step()] both modify their app contexts for the meaning of "modify" in this paragraph.
**
** ^When [system_prepare_v2()] is used to prepare a statement, the statement might be re-prepared during [system_step()] due to a 
** schema change.  Hence, the application should ensure that the correct authorizer callback remains in place during the [system_step()].
**
** ^Note that the authorizer callback is invoked only during [system_prepare()] or its variants.  Authorization is not
** performed during statement evaluation in [system_step()], unless as stated in the previous paragraph, system_step() invokes
** system_prepare_v2() to reprepare a statement after a schema change.
*/
SYSTEM_API int system_set_authorizer(appContext*, int (*xAuth)(void*,int,const char*,const char*,const char*,const char*), void *pUserData);

/*
** API: Authorizer Return Codes
**
** The [system_set_authorizer | authorizer callback function] must return either [SYSTEM_OK] or one of these two constants in order
** to signal APPID whether or not the action is permitted.  See the [system_set_authorizer | authorizer documentation] for additional
** information.
*/
#define SYSTEM_DENY   1   /* Abort the SQL statement with an error */
#define SYSTEM_IGNORE 2   /* Don't allow access, but don't generate an error */

/*
** API: Authorizer Action Codes
**
** The [system_set_authorizer()] interface registers a callback function that is invoked to authorize certain SQL statement actions.  The
** second parameter to the callback is an integer code that specifies what action is being authorized.  These are the integer action codes that
** the authorizer callback may be passed.
**
** These action code values signify what kind of operation is to be authorized.  The 3rd and 4th parameters to the authorization
** callback function will be parameters or NULL depending on which of these codes is used as the second parameter.  ^(The 5th parameter to the
** authorizer callback is the name of the database ("main", "temp", etc.) if applicable.)^  ^The 6th parameter to the authorizer callback
** is the name of the inner-most trigger or view that is responsible for the access attempt or NULL if this access attempt is directly from
** top-level SQL code.
*/
/******************************************* 3rd ************ 4th ***********/
#define SYSTEM_LIMITACTION_CREATE_INDEX          1   /* Index Name      Table Name      */
#define SYSTEM_LIMITACTION_CREATE_TABLE          2   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_CREATE_TEMP_INDEX     3   /* Index Name      Table Name      */
#define SYSTEM_LIMITACTION_CREATE_TEMP_TABLE     4   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_CREATE_TEMP_TRIGGER   5   /* Trigger Name    Table Name      */
#define SYSTEM_LIMITACTION_CREATE_TEMP_VIEW      6   /* View Name       NULL            */
#define SYSTEM_LIMITACTION_CREATE_TRIGGER        7   /* Trigger Name    Table Name      */
#define SYSTEM_LIMITACTION_CREATE_VIEW           8   /* View Name       NULL            */
#define SYSTEM_LIMITACTION_DELETE                9   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_DROP_INDEX           10   /* Index Name      Table Name      */
#define SYSTEM_LIMITACTION_DROP_TABLE           11   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_DROP_TEMP_INDEX      12   /* Index Name      Table Name      */
#define SYSTEM_LIMITACTION_DROP_TEMP_TABLE      13   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_DROP_TEMP_TRIGGER    14   /* Trigger Name    Table Name      */
#define SYSTEM_LIMITACTION_DROP_TEMP_VIEW       15   /* View Name       NULL            */
#define SYSTEM_LIMITACTION_DROP_TRIGGER         16   /* Trigger Name    Table Name      */
#define SYSTEM_LIMITACTION_DROP_VIEW            17   /* View Name       NULL            */
#define SYSTEM_LIMITACTION_INSERT               18   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_PRAGMA               19   /* Pragma Name     1st arg or NULL */
#define SYSTEM_LIMITACTION_READ                 20   /* Table Name      Column Name     */
#define SYSTEM_LIMITACTION_SELECT               21   /* NULL            NULL            */
#define SYSTEM_LIMITACTION_TRANSACTION          22   /* Operation       NULL            */
#define SYSTEM_LIMITACTION_UPDATE               23   /* Table Name      Column Name     */
#define SYSTEM_LIMITACTION_ATTACH               24   /* Filename        NULL            */
#define SYSTEM_LIMITACTION_DETACH               25   /* Database Name   NULL            */
#define SYSTEM_LIMITACTION_ALTER_TABLE          26   /* Database Name   Table Name      */
#define SYSTEM_LIMITACTION_REINDEX              27   /* Index Name      NULL            */
#define SYSTEM_LIMITACTION_ANALYZE              28   /* Table Name      NULL            */
#define SYSTEM_LIMITACTION_CREATE_VTABLE        29   /* Table Name      Module Name     */
#define SYSTEM_LIMITACTION_DROP_VTABLE          30   /* Table Name      Module Name     */
#define SYSTEM_LIMITACTION_FUNCTION             31   /* NULL            Function Name   */
#define SYSTEM_LIMITACTION_SAVEPOINT            32   /* Operation       Savepoint Name  */
#define SYSTEM_LIMITACTION_COPY                  0   /* No longer used */


/*
** API: Run-time Limits
**
** ^(This interface allows the size of various constructs to be limited on a connection by connection basis.  The first parameter is the
** [app context] whose limit is to be set or queried.  The second parameter is one of the [limit categories] that define a
** class of constructs to be size limited.  The third parameter is the new limit for that construct.)^
**
** ^If the new limit is a negative number, the limit is unchanged. ^(For each limit category SYSTEM_LIMIT_<i>NAME</i> there is a 
** [limits | hard upper bound] set at compile-time by a C preprocessor macro called
** [limits | SYSTEM_MAX_<i>NAME</i>]. (The "_LIMIT_" in the name is changed to "_MAX_".))^
** ^Attempts to increase a limit above its hard upper bound are silently truncated to the hard upper bound.
**
** ^Regardless of whether or not the limit was changed, the [system_limit()] interface returns the prior value of the limit.
** ^Hence, to find the current value of a limit without changing it, simply invoke this interface with the third parameter set to -1.
**
** Run-time limits are intended for use in applications that manage both their own internal database and also databases that are controlled
** by untrusted external sources.  An example application might be a web browser that has its own databases for storing history and
** separate databases controlled by JavaScript applications downloaded off the Internet. The internal databases can be given the
** large, default limits.  Databases managed by external sources can be given much smaller limits designed to prevent a denial of service
** attack.  Developers might also want to use the [system_set_authorizer()] interface to further control untrusted SQL.  The size of the database
** created by an untrusted script can be contained using the [max_page_count] [PRAGMA].
**
** New run-time limit categories may be added in future releases.
*/
SYSTEM_API int system_limit(appContext*, int id, int newVal);

/*
** API: Run-Time Limit Categories
** KEYWORDS: {limit category} {*limit categories}
**
** These constants define various performance limits that can be lowered at run-time using [system_limit()].
** The synopsis of the meanings of the various limits is shown below. Additional information is available at [limits | Limits in APPID].
**
** <dl>
** ^(<dt>SYSTEM_LIMIT_LENGTH</dt>
** <dd>The maximum size of any string or BLOB or table row, in bytes.<dd>)^
**
** ^(<dt>SYSTEM_LIMIT_SQL_LENGTH</dt>
** <dd>The maximum length of an SQL statement, in bytes.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_COLUMN</dt>
** <dd>The maximum number of columns in a table definition or in the result set of a [SELECT] or the maximum number of columns in an index
** or in an ORDER BY or GROUP BY clause.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_EXPR_DEPTH</dt>
** <dd>The maximum depth of the parse tree on any expression.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_COMPOUND_SELECT</dt>
** <dd>The maximum number of terms in a compound SELECT statement.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_VDBE_OP</dt>
** <dd>The maximum number of instructions in a virtual machine program used to implement an SQL statement.  This limit is not currently
** enforced, though that might be added in some future release of APPID.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_FUNCTION_ARG</dt>
** <dd>The maximum number of arguments on a function.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_ATTACHED</dt>
** <dd>The maximum number of [ATTACH | attached databases].)^</dd>
**
** ^(<dt>SYSTEM_LIMIT_LIKE_PATTERN_LENGTH</dt>
** <dd>The maximum length of the pattern argument to the [LIKE] or [GLOB] operators.</dd>)^
**
** ^(<dt>SYSTEM_LIMIT_VARIABLE_NUMBER</dt>
** <dd>The maximum index number of any [parameter] in an SQL statement.)^
**
** ^(<dt>SYSTEM_LIMIT_TRIGGER_DEPTH</dt>
** <dd>The maximum depth of recursion for triggers.</dd>)^
** </dl>
*/
#define SYSTEM_LIMIT_LENGTH                    0
#define SYSTEM_LIMIT_SQL_LENGTH                1
#define SYSTEM_LIMIT_COLUMN                    2
#define SYSTEM_LIMIT_EXPR_DEPTH                3
#define SYSTEM_LIMIT_COMPOUND_SELECT           4
#define SYSTEM_LIMIT_VDBE_OP                   5
#define SYSTEM_LIMIT_FUNCTION_ARG              6
#define SYSTEM_LIMIT_ATTACHED                  7
#define SYSTEM_LIMIT_LIKE_PATTERN_LENGTH       8
#define SYSTEM_LIMIT_VARIABLE_NUMBER           9
#define SYSTEM_LIMIT_TRIGGER_DEPTH            10

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_LIMITS_H_ */
