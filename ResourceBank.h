#pragma once
#ifndef RESOURCE_BANK_H
#define RESOURCE_BANK_H
#include "Shader.h"
#include "VertexArray.h"
#include <unordered_map>

namespace ShaderBank
{
	Shader& Get(const std::string& name);
}

namespace VAOBank
{
	VAO& Get(const std::string& name);
}

#endif // RESOURCE_BANK_H