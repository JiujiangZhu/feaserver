// http://www.guineacode.com/2010/linking-and-compiling-opencl/
#include <stdlib.h>
#include "CL/cl.h"


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


const char *sProgramSource = "\
__kernel void \
vectorAdd(__global const float *a, __global const float *b, __global float *c) \
{ \
	// Vector element index \
	int nIndex = get_global_id(0); \
	c[nIndex] = a[nIndex] + b[nIndex]; \
}";

int main()
{
	const unsigned int cnBlockSize = 512;
	const unsigned int cnBlocks = 3;
	const unsigned int cnDimension = cnBlocks * cnBlockSize;

	// create OpenCL device & context
	cl_context hContext;
	hContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, 0, 0, 0);

	// query all devices available to the context
	size_t nContextDescriptorSize;
	clGetContextInfo(hContext, CL_CONTEXT_DEVICES, 0, 0, &nContextDescriptorSize);
	cl_device_id *aDevices = malloc(nContextDescriptorSize);
	clGetContextInfo(hContext, CL_CONTEXT_DEVICES,
	nContextDescriptorSize, aDevices, 0);

	// create a command queue for first device the context reported
	cl_command_queue hCmdQueue;
	hCmdQueue = clCreateCommandQueue(hContext, aDevices[0], 0, 0);

	// create & compile program
	cl_program hProgram;
	hProgram = clCreateProgramWithSource(hContext, 1, &sProgramSource, 0, 0);
	clBuildProgram(hProgram, 0, 0, 0, 0, 0);

	// create kernel
	cl_kernel hKernel;
	hKernel = clCreateKernel(hProgram, "vectorAdd", 0);

	// allocate host vectors
	float *pA = new float[cnDimension];
	float *pB = new float[cnDimension];
	float *pC = new float[cnDimension];

	// initialize host memory
	//randomInit(pA, cnDimension);
	//randomInit(pB, cnDimension);

	// allocate device memory
	cl_mem hDeviceMemA, hDeviceMemB, hDeviceMemC;
	hDeviceMemA = clCreateBuffer(hContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cnDimension * sizeof(cl_float), pA, 0);
	hDeviceMemB = clCreateBuffer(hContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cnDimension * sizeof(cl_float), pA, 0);
	hDeviceMemC = clCreateBuffer(hContext, CL_MEM_WRITE_ONLY, cnDimension * sizeof(cl_float), 0, 0);

	// setup parameter values
	clSetKernelArg(hKernel, 0, sizeof(cl_mem), (void *)&hDeviceMemA);
	clSetKernelArg(hKernel, 1, sizeof(cl_mem), (void *)&hDeviceMemB);
	clSetKernelArg(hKernel, 2, sizeof(cl_mem), (void *)&hDeviceMemC);

	// execute kernel
	clEnqueueNDRangeKernel(hCmdQueue, hKernel, 1, 0, &cnDimension, 0, 0, 0, 0);

	// copy results from device back to host
	clEnqueueReadBuffer(hContext, hDeviceMemC, CL_TRUE, 0, cnDimension * sizeof(cl_float), pC, 0, 0, 0);

	delete[] pA;
	delete[] pB;
	delete[] pC;

	clReleaseMemObj(hDeviceMemA);
	clReleaseMemObj(hDeviceMemB);
	clReleaseMemObj(hDeviceMemC);
}