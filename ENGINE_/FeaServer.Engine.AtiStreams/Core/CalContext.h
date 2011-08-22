#pragma once
#include "cal.h"
#include <stdlib.h>
#include <string>
namespace TimeServices { namespace Engine { namespace Core {
	class CalContext
	{
	private:
		CALcontext _context;
		unsigned int _deviceId;
		CALdevice _device;

	public:
		CalContext(unsigned int deviceId)
			: _deviceId(deviceId), _context(0), _device(0) { }
		bool Initialize();
		bool GetAttributes(CALdeviceattribs& attribs);
		CALcontext GetApiContext() { return _context; }
		void Dispose()
		{
			// Destroy the context
			if (_context)
				if (calCtxDestroy(_context) != CAL_RESULT_OK)
					fprintf(stderr, "Error string is %s\n", calGetErrorString());
			// Close the device
			if (_device)
				if (calDeviceClose(_device) != CAL_RESULT_OK)
				{
					fprintf(stderr, "There was an error closing the device.\n");
					fprintf(stderr, "Error string is %s\n", calGetErrorString());
				}
		}
	};

	static CalContext s_context(0);
}}}
