#pragma once

size_t fallocHeapInBytes(unsigned __int32 totalChunks);
void fallocHeapInitialize(void* heap, unsigned __int32 totalChunks);
void* fallocHeapGet();
void fallocHeapFree(void* obj);

//typedef struct FallocCtx_t FallocCtx;
//FallocCtx* finit();
//void ffree(AllocCtx *t);
//void* falloc(AllocCtx* t, unsigned __int16 bytes, bool zeroAlloc);
