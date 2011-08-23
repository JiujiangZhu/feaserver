#pragma once
#include <cuda.h>
using namespace System;
namespace FeaServer { namespace Engine {
	class CuContext
	{
	private:
		CUcontext _context;
		CUdevice _device;

	public:
		CuContext()
			: _context(0), _device(0) { }

		bool Initialize()
		{
			int deviceCount;
			if ((cuInit(0) != CUDA_SUCCESS)
				|| (cuDeviceGetCount(&deviceCount) != CUDA_SUCCESS)
				|| (deviceCount < 1)
				|| (!GetTightestDevice(deviceCount, _device)))
				return false;
			if (cuCtxCreate(&_context, 0, _device) != CUDA_SUCCESS)
				return false;
			return true;
		}

		static bool GetTightestDevice(int deviceCount, CUdevice &device)
		{
			if (cuDeviceGet(&device, 0) != CUDA_SUCCESS)
				return false;
			return true;
		}

		void Dispose()
		{
			if (_context)
				cuCtxDetach(_context);
		}
	};
}}
