#pragma once
#ifndef GL_DATA_H
#define GL_DATA_H
#include <glew.h>
#include <map>
#include <string>
#include "Shader.h"

struct GLData
{
	struct Data
	{
		GLint type, location;
	};
	std::map<std::string, Data> mapping;

	static GLData GenerateMapping(Shader& shader, GLuint uniformIndex);
};
#endif // GL_DATA_H