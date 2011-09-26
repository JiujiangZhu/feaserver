#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define trace(type,method,...) printf(#type":"method"\n",__VA_ARGS__)
#define thrownew(type,...) throw;