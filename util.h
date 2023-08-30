#pragma once
#ifndef UTIL_H
#define UTIL_H
#include "glmHelp.h"

// TODO: Uh this might be a terrible idea
#define EPSILON std::numeric_limits<float>::epsilon() * 10.f
#define D_EPSILON std::numeric_limits<double>::epsilon() * 10.f

inline glm::vec3 SlideAlongPlane(const glm::vec3& plane, const glm::vec3& direction) noexcept;


inline glm::vec3 SlideAlongPlane(const glm::vec3& plane, const glm::vec3& direction) noexcept
{
	return glm::normalize(direction - glm::dot(direction, plane) * plane);
}

#endif // UTIL_H