#include "CalContext.h"
#include <stdio.h>
#include <string>
namespace TimeServices { namespace Engine { namespace Core {
	bool CalContext::Initialize()
	{
		if (calInit() != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error initializing CAL.\n");
			fprintf(stderr, "Error string is %s\n", calGetErrorString());
			return false;
		}
		// Open the first device
		if (calDeviceOpen(&_device, _deviceId) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error opening the device %d.\n", _deviceId);
			fprintf(stderr, "Error string is %s\n", calGetErrorString());
			return false;
		}
		// Create a CAL context
		if (calCtxCreate(&_context, _device) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error creatint the context.\n");
			fprintf(stderr, "Error string is %s\n", calGetErrorString());
			return false;
		}
		return true;
	}

	bool CalContext::GetAttributes(CALdeviceattribs& attribs)
	{
		attribs.struct_size = sizeof(CALdeviceattribs);
		if (calDeviceGetAttribs(&attribs, _deviceId) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error getting device attribs.\n");
			fprintf(stderr, "Error string is %s\n", calGetErrorString());
			return false;
		}
		return true;
	}

	//CALint QueryDeviceCaps(CALuint DeviceNum) //, SampleFeatures *FeatureList)
	//{
	//	CALboolean capable = CAL_TRUE;
	//	// Get device attributes
	//	CALdeviceattribs attribs;
	//	attribs.struct_size = sizeof(CALdeviceattribs);
	//	if (calDeviceGetAttribs(&attribs, DeviceNum) != CAL_RESULT_OK)
	//	{
	//		fprintf(stderr, "Could not get device attributes.\n");
	//		capable = CAL_FALSE;
	//		return capable;
	//	}
	//	//// Check for requested features
	//	//if (FeatureList->DoublePrecision == CAL_TRUE)
	//	//	if (!attribs.doublePrecision)
	//	//		capable = CAL_FALSE;
	//	//if (FeatureList->ComputeShaders == CAL_TRUE)
	//	//	if (!attribs.computeShader)
	//	//		capable = CAL_FALSE;
	//	//if (FeatureList->LocalDataShares == CAL_TRUE)
	//	//	if (!attribs.localDataShare)
	//	//		capable = CAL_FALSE;
	//	//if (FeatureList->GlobalDataShares == CAL_TRUE)
	//	//	if (!attribs.globalDataShare)
	//	//		capable = CAL_FALSE;
	//	//if (FeatureList->MemExport == CAL_TRUE)
	//	//	if (!attribs.memExport)
	//	//		capable = CAL_FALSE;
	//	return capable;
	//}
}}}