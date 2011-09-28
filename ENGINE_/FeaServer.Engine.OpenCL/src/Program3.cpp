#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
// http://www.guineacode.com/2010/linking-and-compiling-opencl/
#include <stdlib.h>
#include "Core.h"

//int xmain(int argc, char **argv)
//{
//    cl_platform_id test;
//    cl_uint num;
//    cl_uint ok = 1;
//    clGetPlatformIDs(ok, &test, &num);
//
//    return 0;
//}

//
//int main()
//{
//	// create a compute context with GPU device
//	context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
// 
//	// create a command queue
//	queue = clCreateCommandQueue(context, NULL, 0, NULL);
// 
//	// allocate the buffer memory objects
//	memobjs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*2*num_entries, srcA, NULL);
//	memobjs[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*2*num_entries, NULL, NULL);
// 
//	// create the compute program
//	program = clCreateProgramWithSource(context, 1, &fft1D_1024_kernel_src, NULL, NULL);
// 
//	// build the compute program executable
//	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
// 
//	// create the compute kernel
//	kernel = clCreateKernel(program, "fft1D_1024", NULL);
// 
//	// set the args values
//	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjs[0]);
//	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjs[1]);
//	clSetKernelArg(kernel, 2, sizeof(float)*(local_work_size[0]+1)*16, NULL);
//	clSetKernelArg(kernel, 3, sizeof(float)*(local_work_size[0]+1)*16, NULL);
// 
//	// create N-D range object with work-item dimensions and execute kernel
//	global_work_size[0] = num_entries;
//	local_work_size[0] = 64;
//	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
//}
