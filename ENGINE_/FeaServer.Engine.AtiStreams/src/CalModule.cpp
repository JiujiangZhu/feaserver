#include "calcl.h"
#include "CalContext.h"
#include "CalModule.h"
#include <stdio.h>
#include <string>
namespace TimeServices { namespace Engine { namespace Core {
	static void __logger(const CALchar *text)
	{
		fprintf(stderr, text);
	}

	bool CalModule::Initialize(const CALchar* ilKernel, bool disassemble)
	{
		CALimage image = NULL;
		CALboolean success = CAL_FALSE;

		// Get device specific information
		CALdeviceattribs attribs;
		if (!_context->GetAttributes(attribs))
			return false;

		// Compile IL kernel into object
		CALobject obj;
		if (calclCompile(&obj, CAL_LANGUAGE_IL, ilKernel, attribs.target) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error compiling the program.\n");
			fprintf(stderr, "Error string is %s\n", calclGetErrorString());
			return false;
		}

		// Link object into an image
		if (calclLink(&image, &obj, 1) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error linking the program.\n");
			fprintf(stderr, "Error string is %s\n", calclGetErrorString());
			return false;
		}
		if (calclFreeObject(obj) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error freeing the compiler object.\n");
			fprintf(stderr, "Error string: %s\n", calclGetErrorString());
			return false;
		}
		if (disassemble == true)
			calclDisassembleImage(image, __logger);

		// Load module into the context
		if (calModuleLoad(&_module, _context->GetApiContext(), image) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error loading the program module.\n");
			fprintf(stderr, "Error string is %s\n", calGetErrorString());
			return false;
		}
		if (calclFreeImage(image) != CAL_RESULT_OK)
		{
			fprintf(stderr, "There was an error freeing the program image.\n");
			fprintf(stderr, "Error string is %s\n", calGetErrorString());
			return false;
		}
		return true;
	}
}}}