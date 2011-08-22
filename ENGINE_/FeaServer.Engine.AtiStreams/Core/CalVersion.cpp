#include "CalVersion.h"
#include <stdio.h>
#include <string>
namespace TimeServices { namespace Engine { namespace Core {
	CALint CalVersion::QueryVersion(const CALchar* comparison)
	{
		CalVersion available;
		available.GetVersion();
		printf("Found CAL Runtime Version: %d.%d.%d\n", available.major, available.minor, available.imp);
		if (strcmp(comparison, ">") == 0)
		{
			if ((available.major > major) ||
				((available.major == major) && (available.minor > minor)) ||
				((available.major == major) && (available.minor == minor) &&
				(available.imp > imp)))
				return 1;
		}
		else if (strcmp(comparison, ">=") == 0)
		{
			if ((available.major > major) ||
				((available.major == major) && (available.minor > minor)) ||
				((available.major == major) && (available.minor == minor) && (available.imp >= imp)))
				return 1;
		}
		else if (strcmp(comparison, "<") == 0)
		{
			if ((available.major < major) ||
				((available.major == major) && (available.minor < minor)) ||
				((available.major == major) && (available.minor == minor) && (available.imp < imp)))
				return 1;
		}
		else if (strcmp(comparison, "<=") == 0)
		{
			if ((available.major < major) ||
				((available.major == major) && (available.minor < minor)) ||
				((available.major == major) && (available.minor == minor) && (available.imp <= imp)))
				return 1;
		}
		else if (strcmp(comparison, "==") == 0)
		{
			if ((available.major == major) && (available.minor == minor) && (available.imp == imp))
				return 1;
		}
		else 
			fprintf(stderr, "Error. Invalid comparison operator: %s (QueryCALVersion)\n", comparison);
		return 0;
	}
}}}
