#ifndef CLPRINTF_CL
#define CLPRINTF_CL
#include "clPrintf.clh"

#define CLPRINTF_MAXLEN 256

// This structure is used internally to track block/thread output restrictions.
typedef struct __attribute__((aligned (8))) {
	int globalId;
} clPrintfRestriction;

// This structure is used internally to track block/thread output restrictions.
struct __attribute__((aligned (8))) _clPrintfContext {
	int* globalPrintfBuffer;
	int printfBufferLength;
	clPrintfRestriction restrictRules;
	int* printfBufferPtr;
};

typedef struct __attribute__((aligned (8))) {
    unsigned int magic;
    unsigned int fmtoffset;
    int globalId;
} clPrintfHeader;

#define CLPRINTF_ALIGNSIZE 8
#define CLPRINTF_MAGIC (unsigned short)0xC811

int* getNextPrintfBufPtr(__global clPrintfContext* ctx)
{
    if (!ctx)
        return nullptr;
	// thread/block restriction check
	clPrintfRestriction restrictRules = ctx->restrictRules;
	if ((restrictRules.globalId != CLPRINTF_UNRESTRICTED) && (restrictRules.globalId != get_global_id(0)))
		return nullptr;
	// much easier with an atomic operation!
	size_t offset = (size_t)ctx->printfBufferPtr - (size_t)ctx->globalPrintfBuffer;
	ctx->printfBufferPtr += CLPRINTF_MAXLEN;
	offset %= ctx->printfBufferLength;
    return ctx->globalPrintfBuffer + offset;
}

void writePrintfHeader(int* ptr, int* fmtptr)
{
    if (ptr) {
        clPrintfHeader header;
        header.magic = CLPRINTF_MAGIC;
        header.fmtoffset = (unsigned int)(fmtptr - ptr);
        header.globalId = get_global_id(0);
        *((clPrintfHeader*)ptr) = header;
    }
}

int* clPrintfStrncpy(int* dest, __constant int* src, int n, int* end)
{
    if (!dest || !src || (dest >= end))
        return nullptr;
    //
    int* lenptr = dest;
    int len = 0;
    dest += CLPRINTF_ALIGNSIZE;
    // Now copy the string
    while (n--) {
        if (dest >= end)
            break;
        len++;
        *dest++ = *src;
        if (*src++ == 0)
            break;
    }
    // Now write out the padding bytes, and we have our length.
    while ((dest < end) && (((size_t)dest & (CLPRINTF_ALIGNSIZE-1)) != 0)) { len++; *dest++ = 0; }
    *lenptr = len;
    return (dest < end ? dest : nullptr);
}

int* copyArg(int* ptr, __constant int* arg, int* end)
{
    if (!ptr || !arg)
        return nullptr;
    // strncpy does all our work. We just terminate.
    if ((ptr = clPrintfStrncpy(ptr, arg, CLPRINTF_MAXLEN, end)) != nullptr)
        *ptr = nullptr;
    return ptr;
}

#define CLPRINTF_PREAMBLE \
    int *start, *end, *bufptr, *fmtstart; \
    if ((start = getNextPrintfBufPtr(ctx)) == nullptr) return 0; \
    end = start + CLPRINTF_MAXLEN; \
    bufptr = start + sizeof(clPrintfHeader);

// Posting an argument is easy
#define CLPRINTF_ARG(argname) \
	bufptr = copyArg(bufptr, argname, end);

// After args are done, record start-of-fmt and write the fmt and header
#define CLPRINTF_POSTAMBLE \
    fmtstart = bufptr; \
    end = clPrintfStrncpy(bufptr, fmt, CLPRINTF_MAXLEN, end); \
    writePrintfHeader(start, end ? fmtstart : nullptr); \
    return (end ? (int)(end - start) : 0);

int clPrintf(__global clPrintfContext* ctx, __constant int* fmt) {
	CLPRINTF_PREAMBLE;
	CLPRINTF_POSTAMBLE;
}

#endif
