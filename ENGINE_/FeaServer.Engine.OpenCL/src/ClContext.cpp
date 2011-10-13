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
using namespace System::Runtime::InteropServices;
using namespace FeaServer::Engine::Utility;
// http://www.guineacode.com/2010/linking-and-compiling-opencl/
// http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetProgramBuildInfo.html
// http://www.cmsoft.com.br/index.php?option=com_content&view=category&layout=blog&id=113&Itemid=168

namespace FeaServer { namespace Engine {
	class ClContext
	{
	public:
		cl_device_id _device_id;
		cl_context _context;
		cl_command_queue _command_queue;

	public:
		ClContext()
			: _context(nullptr)  { }

		static void ContextNotifyHandler(const char* errinfo, const void* private_info, size_t cb, void* user_data)
		{
			printf("ContextNotifyHandler");
		}

		bool Initialize()
		{
			cl_int r;

			// create OpenCL device
			cl_platform_id platform_id;
			cl_uint ret_num_platforms;
			r = clGetPlatformIDs(1, &platform_id, &ret_num_platforms); assertR(r, "Exception", "clGetPlatformIDs");
			cl_uint ret_num_devices;
			r = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &_device_id, &ret_num_devices); assertR(r, "Exception", "clGetDeviceIDs");

			// create OpenCL context
			_context = clCreateContext(nullptr, 1, &_device_id, ContextNotifyHandler, nullptr, &r); assertR(r, "Exception", "clCreateContext");

			// create a command queue for first device the context reported
			_command_queue = clCreateCommandQueue(_context, _device_id, 0, &r); assertR(r, "Exception", "clCreateCommandQueue");

			return true;
		}

		void Dispose()
		{
		}

		cl_program CreateProgram(String^ pathMngd)
		{
			String^ bodyMngd = Preparser::ReadAllText(pathMngd);
			// create & compile program
			cl_int r;
			const size_t source_size = 0;
			char* body = (char*)Marshal::StringToHGlobalAnsi(bodyMngd).ToPointer();
			cl_program program = clCreateProgramWithSource(_context, 1, (const char**)&body, 0, &r);
			Marshal::FreeHGlobal(IntPtr(body));
			assertR(r, "Exception", "clCreateProgramWithSource");
			r = clBuildProgram(program, 1, &_device_id, "-Werror", nullptr, nullptr);
			if (r != CL_SUCCESS) {
				char log[10240];
				r = clGetProgramBuildInfo(program, _device_id, CL_PROGRAM_BUILD_LOG, 10240, log, NULL);
				printf("COMPILE ERROR:\n");
				if (r == CL_SUCCESS)
					printf("%s\n", log);
				else
					printf("Error was too long.");
				scanf_s("%c"); exit(0);
			}
			return program;
		}
	};
}}
