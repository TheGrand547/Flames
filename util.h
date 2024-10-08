#pragma once
#ifndef UTIL_H
#define UTIL_H
#include "glmHelp.h"

// TODO: Uh this might be a terrible idea
#define EPSILON std::numeric_limits<float>::epsilon() * 1000.f
#define D_EPSILON std::numeric_limits<double>::epsilon() * 1000.f

// Has to be zero for some reason
#define BORDER_PARAMETER 0

namespace Tick
{
	constexpr float TimeDelta = 0x1.p-7f;
}

inline glm::vec3 SlideAlongPlane(const glm::vec3& plane, const glm::vec3& direction) noexcept
{
	return glm::normalize(direction - glm::dot(direction, plane) * plane);
}

glm::vec2 GetProjectionHalfs(glm::mat4& mat);

#endif // UTIL_H