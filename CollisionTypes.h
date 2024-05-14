#pragma once
#ifndef COLLISION_TYPES_H
#define COLLISION_TYPES_H
#include <glm/glm.hpp>
#include <iostream>
#include "glmHelp.h"

/*
 * For a function A.Overlaps(B) or A.Intersection(B): A will be treated a static entity, B will be a moveable one
 * 
 * the resulting Collision struct will have the following format
 *      point will be the center of B which is closest to A without a collision occuring 
 *      normal is the direction in which the collision takes place
 *      depth is the amount of intersection in the direction of normal
 */
struct Collision
{
	union
	{
		glm::vec3 point, center;
	};

	union
	{
		glm::vec3 normal, axis;
	};

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

struct RotationCollision : public Collision {};
struct SlidingCollision : public Collision {};
struct RayCollision : public Collision {};

inline std::ostream& operator<<(std::ostream& os, const Collision& collision) noexcept
{
	os << collision.point << "\t" << collision.normal << "\t" << collision.depth;
	return os;
}
#endif // COLLISION_TYPES_H