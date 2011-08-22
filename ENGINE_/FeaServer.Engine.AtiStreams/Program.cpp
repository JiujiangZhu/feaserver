#include "Program.h"
#include "Core\CalVersion.h"
#include "Core\CalContext.h"
#include "Core\CalModule.h"
#include <stdio.h>
#include <string>
using namespace Core;

int main(CALint argc, CALchar** argv)
{
	CalVersion calVersion = { 1, 4, 519 };
	printf("Supported CAL Runtime Version: %d.%d.%d\n", calVersion.major, calVersion.minor, calVersion.imp);

	// Validate the available CAL runtime version 
	if (!calVersion.QueryVersion())
	{
		fprintf(stdout, "Error. Could not find a compatible CAL runtime.\n");
		return 0;
	}
	
    // Initializing CAL and opening device
	CalContext context(0);
    if (!context.Initialize())
        return 1;

std::string programIL =
    "il_ps_2_0                                                                              \n"
    "dcl_input_interp(linear) v0.xy__                                                       \n"
    "dcl_output_generic o0                                                                  \n"
    "dcl_cb cb0[1]                                                                          \n"
    "dcl_resource_id(0)_type(2d,unnorm)_fmtx(float)_fmty(float)_fmtz(float)_fmtw(float)     \n"
    "sample_resource(0)_sampler(0) r0, v0.xyxx                                              \n"
    "mul o0, r0, cb0[0]                                                                     \n"
    "end                                                                                    \n";

	CalModule module(context);
	module.Initialize(programIL.c_str(), true);

	//
	fprintf(stdout, "Done.\n");
	return 0;
}
