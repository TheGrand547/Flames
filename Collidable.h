#pragma once
#ifndef COLLIDABLE_H
#define COLLIDABLE_H
#include <glm/glm.hpp>
#include "AABB.h"

struct Collision
{
	glm::vec3 point, normal;
	union
	{
		float distance, depth;
	};

	Collision() = default;
	inline consteval Collision(const glm::vec3& point, const glm::vec3& normal, const float& distance) noexcept 
		: point(point), normal(normal), distance(distance) {}
	~Collision() = default;
	consteval bool operator==(const Collision& other) const = default;
	consteval bool operator!=(const Collision& other) const = default;
	
	inline constexpr void Clear()
	{
		this->point = glm::vec3(0);
		this->normal = glm::vec3(0);
		this->distance = 0;
	}

	Collision& operator=(const Collision& other) = default;
};

class Collidable
{
	virtual bool Collide() const = 0;
	virtual AABB GetBoundingBox() const = 0;
};

#endif // COLLIDABLE_H