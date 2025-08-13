#pragma once
#ifndef TRANSFORM_H
#define TRANSFORM_H
#include "glmHelp.h"

struct Transform
{
	glm::vec3 position{ 0 };
	glm::quat rotation{glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 0.f, 0.f)};
	
	inline Transform(const glm::vec3& position = glm::vec3(0), const glm::quat& rotation = 
		glm::quat(glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 0.f, 0.f))) noexcept : position(position), rotation(rotation) {}

	inline void Normalize() noexcept;
	inline Transform Normalized() const noexcept;

	// Apply the parent transform to this, then return the resulting transformation
	inline Transform Append(const Transform& other) const noexcept;

	// Apply the parent transform to this, interpreting *this.position as a global offset
	inline Transform AppendWorld(const Transform& other) const noexcept;
};

inline void Transform::Normalize() noexcept
{
	this->rotation = glm::normalize(this->rotation);
}

inline Transform Transform::Normalized() const noexcept
{
	return { this->position, glm::normalize(this->rotation) };
}

inline Transform Transform::Append(const Transform& parent) const noexcept
{
	Transform temp{ *this };
	temp.rotation = parent.rotation * temp.rotation;
	temp.position = parent.position + temp.rotation * temp.position;
	return temp;
}

inline Transform Transform::AppendWorld(const Transform& parent) const noexcept
{
	Transform temp{ *this };
	temp.rotation = parent.rotation * temp.rotation;
	temp.position = parent.position + temp.position;
	return temp;
}

#endif // TRANSFORM_H