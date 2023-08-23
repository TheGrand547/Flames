#pragma once
#ifndef COLLIDABLE_H
#define COLLIDABLE_H
#include <glm/glm.hpp>

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
#endif // COLLIDABLE_H