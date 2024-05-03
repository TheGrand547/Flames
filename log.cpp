#include "log.h"
#include <glew.h>
#include <iostream>

void CheckError(const std::source_location location)
{
	GLenum e;
	while ((e = glGetError()))
	{
		printf("%s OpenGL Error: %s\n", LocationFormat(location).c_str(), reinterpret_cast<const char*>(gluErrorString(e)));
	}
}