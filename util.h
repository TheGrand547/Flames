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

namespace World
{
	const glm::vec3 Forward = glm::vec3(1.f, 0.f, 0.f);
	const glm::vec3 Up      = glm::vec3(0.f, 1.f, 0.f);
	const glm::vec3 Right   = glm::vec3(0.f, 0.f, 1.f);
	const glm::vec3 Zero    = glm::vec3(0.f);
}

inline glm::vec3 SlideAlongPlane(const glm::vec3& plane, const glm::vec3& direction) noexcept
{
	return glm::normalize(direction - glm::dot(direction, plane) * plane);
}

// Ensure that nan doesn't pollute calculations
inline float Rectify(const float& value, const float& reference = 0.f) noexcept
{
	return (!glm::isnan(value)) ? value : reference;
}

glm::vec2 GetProjectionHalfs(glm::mat4& mat);

#endif // UTIL_H