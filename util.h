#pragma once
#ifndef UTIL_H
#define UTIL_H
#include "glmHelp.h"

#define EPSILON 0.000001f

inline glm::vec3 SlideAlongPlane(const glm::vec3& plane, const glm::vec3& direction) noexcept;


inline glm::vec3 SlideAlongPlane(const glm::vec3& plane, const glm::vec3& direction) noexcept
{
	return glm::normalize(direction - glm::dot(direction, plane) * plane);
}

#endif // UTIL_H