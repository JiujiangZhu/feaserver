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
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "Core.h"
using namespace System;
using namespace FeaServer::Engine::Utility;
// http://www.guineacode.com/2010/linking-and-compiling-opencl/

const char *source_str = "\
__kernel void \
vectorAdd(__global const float *a, __global const float *b, __global float *c) \
{ \
	int nIndex = get_global_id(0); \
	c[nIndex] = a[nIndex] + b[nIndex]; \
}";

void ContextNotifyHandler(const char *errinfo, const void *private_info, size_t  cb, void *user_data)
{
}

int xmain()
{
	String^ body = Preparser::ReadAllText("C:\\_APPLICATION\\FEASERVER\\ENGINE_\\FeaServer.Engine.OpenCL\\src\\Time\\SchedulerKernel.cl");

	const unsigned int cnBlockSize = 512;
	const unsigned int cnBlocks = 3;
	const size_t dimension = cnBlocks * cnBlockSize;
	cl_int r;

	// start context
    cl_platform_id platform_id;
	cl_uint ret_num_platforms;
	r = clGetPlatformIDs(1, &platform_id, &ret_num_platforms); assertR(r, "Exception", "clGetPlatformIDs");
	cl_device_id device_id;
	cl_uint ret_num_devices;
    r = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices); assertR(r, "Exception", "clGetDeviceIDs");

	// create OpenCL device & context
	cl_context context = clCreateContext(nullptr, 1, &device_id, ContextNotifyHandler, nullptr, &r); assertR(r, "Exception", "clCreateContext");

	// create a command queue for first device the context reported
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &r); assertR(r, "Exception", "clCreateCommandQueue");

	// allocate host vectors and device memory
	float *a = new float[dimension];
	float *b = new float[dimension];
	float *c = new float[dimension];
	cl_mem deviceMemA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dimension * sizeof(cl_float), a, &r); assertR(r, "Exception", "clCreateBuffer.0");
	cl_mem deviceMemB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dimension * sizeof(cl_float), b, &r); assertR(r, "Exception", "clCreateBuffer.2");
	cl_mem deviceMemC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dimension * sizeof(cl_float), nullptr, &r); assertR(r, "Exception", "clCreateBuffer.2");

	// create & compile program
	const size_t source_size = 0;
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &r); assertR(r, "Exception", "clCreateProgramWithSource");
	r = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr); assertR(r, "Exception", "clBuildProgram");

	// create kernel
	cl_kernel kernel = clCreateKernel(program, "vectorAdd", &r); assertR(r, "Exception", "clCreateKernel");

	// setup parameter values
	r = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&deviceMemA); assertR(r, "Exception", "clSetKernelArg.0");
	r = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&deviceMemB); assertR(r, "Exception", "clSetKernelArg.1");
	r = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&deviceMemC); assertR(r, "Exception", "clSetKernelArg.2");

	// execute kernel
	r = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &dimension, 0, 0, nullptr, nullptr); assertR(r, "Exception", "clEnqueueNDRangeKernel");

	// copy results from device back to host
	r = clEnqueueReadBuffer(command_queue, deviceMemC, CL_TRUE, 0, dimension * sizeof(cl_float), c, 0, nullptr, nullptr); assertR(r, "Exception", "clEnqueueReadBuffer");

	delete[] a;
	delete[] b;
	delete[] c;

	clReleaseMemObject(deviceMemA);
	clReleaseMemObject(deviceMemB);
	clReleaseMemObject(deviceMemC);

	//
	printf("done."); scanf_s("%c");
	return 0;
}
