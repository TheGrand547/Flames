#pragma once
#ifndef SCREEN_SPACE_H
#define SCREEN_SPACE_H
#include <glm/glm.hpp>
#include "Shader.h"
#include "UniformBuffer.h"

namespace ScreenSpace
{
	Shader& GetShader();
	UniformBuffer& GetProjection();
	void Setup();
}
#endif // SCREEN_SPACE_H

