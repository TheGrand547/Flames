#pragma once
#ifndef GLM_HELP_H
#define GLM_HELP_H
#include <format>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/io.hpp>
#include <iostream>
#include <limits>

#define RotateX(matrix, radians) glm::rotate(matrix, radians, glm::vec3(1.f, 0.f, 0.f))
#define RotationX(radians) RotateX(glm::mat4(1.0f), radians)

#define RotateY(matrix, radians) glm::rotate(matrix, radians, glm::vec3(0.f, 1.f, 0.f))
#define RotationY(radians) RotateY(glm::mat4(1.0f), radians)

#define RotateZ(matrix, radians) glm::rotate(matrix, radians, glm::vec3(0.f, 0.f, 1.f))
#define RotationZ(radians) RotateZ(glm::mat4(1.0f), radians)

#define Vec4to3(vec4) glm::vec3(vec4.x, vec4.y, vec4.z)
#define Vec3to4(vec3) glm::vec4(vec3.x, vec3.y, vec3.z, 1.f)

#define absdot(x, y) glm::abs(glm::dot(x, y))

template<typename T> concept IsVec3 = std::same_as<std::remove_cvref_t<T>, glm::vec3>;

glm::mat3 SetForward(const glm::vec3& vec, const glm::vec3& up = glm::vec3(0, 1, 0)) noexcept;

#endif // GLM_HELP_H