#pragma once
#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H
#include <array>
#include <glm/gtc/epsilon.hpp>
#include <glm/glm.hpp>
#include "AABB.h"

// TODO: EPSILON DUMB FUCK
#define EPSILON 0.000001

class OrientedBoundingBox
{
private:
	// These are the basis vectors
	glm::vec3 center; // Maybe center?
	std::array<std::pair<glm::vec3, float>, 3> axes;
public:
	constexpr OrientedBoundingBox();
	constexpr OrientedBoundingBox(const AABB& other);
	~OrientedBoundingBox();

	inline constexpr void Translate(const glm::vec3& distance);
	inline constexpr void Rotate(const glm::vec3& distance);

	constexpr bool Intersect(glm::vec3 point, glm::vec3 dir, float& distance) const;
	constexpr bool Overlap(const OrientedBoundingBox& other) const;
};

inline constexpr void OrientedBoundingBox::Translate(const glm::vec3& distance)
{
	this->center += distance;
}

typedef OrientedBoundingBox OBB;

// Triangles are laid out like (123) (234) (345) in the list, repeated tris
OBB MakeOBB(std::vector<glm::vec3> triangles)
{

}

OBB 

#endif // ORIENTED_BOUNDING_BOX_H
