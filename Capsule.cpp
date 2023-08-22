#include "Capsule.h"

bool Capsule::Intersect(const Capsule& other) const noexcept
{
	Collision temp{};
	return this->Intersect(other, temp);
}

// See: https://wickedengine.net/2020/04/26/capsule-collision-detection/

bool Capsule::Intersect(const Capsule& other, Collision& hit) const noexcept
{
	glm::vec3 bestA(0), bestB(0);
	float distance = this->line.Distance(other.line, bestA, bestB);
	hit.normal = glm::normalize(bestB - bestA);
	hit.distance = this->radius + other.radius - distance; // How far into this capsule the other is
	hit.point = bestB + other.radius * hit.normal; // The point of other furthest into the 
	return distance > 0;
}

bool Capsule::Intersect(const Sphere& other) const noexcept
{
	Collision temp{};
	return this->Intersect(other, temp);
}

bool Capsule::Intersect(const Sphere& other, Collision& hit) const noexcept
{
	glm::vec3 closest = this->line.PointClosestTo(other.center);
	hit.normal = glm::normalize(closest - other.center);
	hit.distance = this->radius + other.radius - glm::length(closest - other.center);
	hit.point = closest + hit.normal * other.radius; 
	return hit.distance > 0;
}

glm::vec3 Capsule::ClosestPoint(const glm::vec3& other) const
{
	return this->line.PointClosestTo(other);
}
