#pragma once
#ifndef TRANSFORM_H
#define TRANSFORM_H
#include "glmHelp.h"

struct Transform
{
	glm::vec3 position{ 0 };
	glm::quat rotation{};
	
	inline Transform(const glm::vec3& position = glm::vec3(0), const glm::quat& rotation = glm::quat()) noexcept
		: position(position), rotation(rotation) {}
};

#endif // TRANSFORM_H