#include "log.h"
#include <iostream>

void CheckErrors(int line, const char* file, const char* function)
{
	GLenum e;
	while ((e = glGetError()))
	{
		std::string given((char*) gluErrorString(e));
		printf("[%s][%s][%i] %s\n", file, function, line, given.c_str());
	}
}