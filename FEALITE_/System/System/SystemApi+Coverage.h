#ifndef _SYSTEMAPI_COVERAGE_H_
#define _SYSTEMAPI_COVERAGE_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Testing Interface
**
** ^The system_test_control() interface is used to read out internal state of APPID and to inject faults into APPID for testing
** purposes.  ^The first parameter is an operation code that determines the number, meaning, and operation of all subsequent parameters.
**
** This interface is not for use by applications.  It exists solely for verifying the correct operation of the API library.  Depending
** on how the APID library is compiled, this interface might not exist.
**
** The details of the operation codes, their meanings, the parameters they take, and what they do are all subject to change without notice.
** Unlike most of the APPID API, this function is not guaranteed to operate consistently from one release to the next.
*/
SYSTEM_API int system_test_control(int op, ...);

/*
** API: Testing Interface Operation Codes
**
** These constants are the valid operation code parameters used as the first argument to [system_test_control()].
**
** These parameters and their meanings are subject to change without notice.  These values are for testing purposes only.
** Applications should not use any of these parameters or the [system_test_control()] interface.
*/
#define SYSTEM_TESTCTRL_FIRST                    5
#define SYSTEM_TESTCTRL_PRNG_SAVE                5
#define SYSTEM_TESTCTRL_PRNG_RESTORE             6
#define SYSTEM_TESTCTRL_PRNG_RESET               7
#define SYSTEM_TESTCTRL_BITVEC_TEST              8
#define SYSTEM_TESTCTRL_FAULT_INSTALL            9
#define SYSTEM_TESTCTRL_BENIGN_MALLOC_HOOKS     10
#define SYSTEM_TESTCTRL_PENDING_BYTE            11
#define SYSTEM_TESTCTRL_ASSERT                  12
#define SYSTEM_TESTCTRL_ALWAYS                  13
#define SYSTEM_TESTCTRL_RESERVE                 14
#define SYSTEM_TESTCTRL_OPTIMIZATIONS           15
#define SYSTEM_TESTCTRL_ISKEYWORD               16
#define SYSTEM_TESTCTRL_PGHDRSZ                 17
#define SYSTEM_TESTCTRL_SCRATCHMALLOC           18
#define SYSTEM_TESTCTRL_LAST                    18

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_COVERAGE_H_ */

