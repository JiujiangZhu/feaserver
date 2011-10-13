template <typename T>
__device__ static char *copyArg(char *ptr, T &arg, char *end)
{
    if (!ptr || ((ptr+CLPRINTF_ALIGNSIZE) >= end))
        return NULL;
    // Write the length and argument
    *(int*)(void*)ptr = sizeof(arg);
    ptr += CLPRINTF_ALIGNSIZE;
    *(T*)(void*)ptr = arg;
    ptr += CLPRINTF_ALIGNSIZE;
    *ptr = 0;
    return ptr;
}

template <typename T1>
__device__ int cuPrintf(const char* fmt, T1 arg1) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_ARG(arg5);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_ARG(arg5);
	CUPRINTF_ARG(arg6);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_ARG(arg5);
	CUPRINTF_ARG(arg6);
	CUPRINTF_ARG(arg7);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_ARG(arg5);
	CUPRINTF_ARG(arg6);
	CUPRINTF_ARG(arg7);
	CUPRINTF_ARG(arg8);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_ARG(arg5);
	CUPRINTF_ARG(arg6);
	CUPRINTF_ARG(arg7);
	CUPRINTF_ARG(arg8);
	CUPRINTF_ARG(arg9);
	CUPRINTF_POSTAMBLE;
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
__device__ int cuPrintf(const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10) {
	CUPRINTF_PREAMBLE;
	CUPRINTF_ARG(arg1);
	CUPRINTF_ARG(arg2);
	CUPRINTF_ARG(arg3);
	CUPRINTF_ARG(arg4);
	CUPRINTF_ARG(arg5);
	CUPRINTF_ARG(arg6);
	CUPRINTF_ARG(arg7);
	CUPRINTF_ARG(arg8);
	CUPRINTF_ARG(arg9);
	CUPRINTF_ARG(arg10);
	CUPRINTF_POSTAMBLE;
}
