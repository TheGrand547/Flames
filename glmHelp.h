#pragma once
#ifndef GLM_HELP_H
#define GLM_HELP_H
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>

#define RotateX(matrix, radians) glm::rotate(matrix, radians, glm::vec3(1.f, 0.f, 0.f))
#define RotationX(radians) RotateX(glm::mat4(1.0f), radians)

#define RotateY(matrix, radians) glm::rotate(matrix, radians, glm::vec3(0.f, 1.f, 0.f))
#define RotationY(radians) RotateY(glm::mat4(1.0f), radians)

#define RotateZ(matrix, radians) glm::rotate(matrix, radians, glm::vec3(0.f, 0.f, 1.f))
#define RotationZ(radians) RotateZ(glm::mat4(1.0f), radians)

#define Vec4to3(vec4) glm::vec3(vec4.x, vec4.y, vec4.z)
#define Vec3to4(vec3) glm::vec4(vec3.x, vec3.y, vec3.z, 1.f)

#define absdot(x, y) glm::abs(glm::dot(x, y))

//std::ostream& operator<<(std::ostream& os, const glm::vec3& vec);

template<glm::length_t T, typename W, glm::qualifier Q>
std::ostream& operator<<(std::ostream& os, const glm::vec<T, W, Q>& vec)
{
	os << "(";
	for (glm::length_t i = 0; i < T; i++)
	{
		os << vec[i];
		if (i != T - 1)
			os << ", ";
	}
	os << ")";
	return os;
}

//  SLOPPY
inline std::ostream& operator<<(std::ostream& os, const glm::quat& vec)
{
	os << "(";
	for (glm::length_t i = 0; i < 4; i++)
	{
		os << vec[i];
		if (i != 4 - 1)
			os << ", ";
	}
	os << ")";
	return os;
}

template<typename T> concept IsVec3 = std::same_as<std::remove_cvref_t<T>, glm::vec3>;

namespace glm 
{
	// Returns the absolute value of the dot product of x on y
	template<length_t L, typename T, qualifier Q> inline T adot(vec<L, T, Q> const& x, vec<L, T, Q> const& y)
	{
		return glm::abs(glm::dot(x, y));
	}
}

#endif // GLM_HELP_H