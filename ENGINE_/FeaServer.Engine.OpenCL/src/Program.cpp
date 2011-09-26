// http://www.guineacode.com/2010/linking-and-compiling-opencl/
#include <stdlib.h>
#include "Core.h"

const char *_programSource = "\
__kernel void \
vectorAdd(__global const float *a, __global const float *b, __global float *c) \
{ \
	// Vector element index \
	int nIndex = get_global_id(0); \
	c[nIndex] = a[nIndex] + b[nIndex]; \
}";

void ContextNotifyHandler(const char *errinfo, const void  *private_info, size_t  cb, void  *user_data)
{
}

int xmain()
{
	const unsigned int cnBlockSize = 512;
	const unsigned int cnBlocks = 3;
	const size_t dimension = cnBlocks * cnBlockSize;

	cl_int r;

	cl_int num_entries;
	cl_platform_id platforms;
	cl_uint num_platforms;
	//clGetPlatformIDs(
	/*
	
    cl_int ret;
	if ((r = clGetPlatformIDs(1, &platform_id, &ret_num_platforms)) != CL_SUCCESS)
		thrownew("Exception", "clGetPlatformIDs");
    if ((r = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices) != CL_SUCCESS)
		thrownew("Exception", "clGetDeviceIDs");
*/

	// create OpenCL device & context
	cl_context context = clCreateContextFromType(nullptr, CL_DEVICE_TYPE_DEFAULT, ContextNotifyHandler, 0, &r);
	if (r == CL_SUCCESS)
		thrownew("Exception", "clCreateContextFromType");

	// query all devices available to the context
	size_t contextDescriptorSize;
	if ((r = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &contextDescriptorSize)) != CL_SUCCESS)
		thrownew("Exception", "clGetContextInfo");
	cl_device_id *devices = (cl_device_id *)malloc(contextDescriptorSize);
	if ((r = clGetContextInfo(context, CL_CONTEXT_DEVICES, contextDescriptorSize, devices, 0)) != CL_SUCCESS)
		thrownew("Exception", "clGetContextInfo2");

	// create a command queue for first device the context reported
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 0, 0);

	// create & compile program
	cl_program program = clCreateProgramWithSource(context, 1, &_programSource, 0, 0);
	if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS)
		thrownew("Exception", "clBuildProgram");

	// create kernel
	cl_kernel kernel = clCreateKernel(program, "vectorAdd", 0);

	// allocate host vectors
	float *a = new float[dimension];
	float *b = new float[dimension];
	float *c = new float[dimension];

	// initialize host memory
	//randomInit(pA, cnDimension);
	//randomInit(pB, cnDimension);

	// allocate device memory
	cl_mem deviceMemA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dimension * sizeof(cl_float), a, 0);
	cl_mem deviceMemB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dimension * sizeof(cl_float), a, 0);
	cl_mem deviceMemC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dimension * sizeof(cl_float), 0, 0);

	// setup parameter values
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&deviceMemA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&deviceMemA);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&deviceMemA);

	// execute kernel
	clEnqueueNDRangeKernel(cmdQueue, kernel, 1, 0, &dimension, 0, 0, 0, 0);

	// copy results from device back to host
	clEnqueueReadBuffer(cmdQueue, deviceMemC, CL_TRUE, 0, dimension * sizeof(cl_float), c, 0, 0, 0);

	delete[] a;
	delete[] b;
	delete[] c;

	clReleaseMemObject(deviceMemA);
	clReleaseMemObject(deviceMemB);
	clReleaseMemObject(deviceMemC);
}
