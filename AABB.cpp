#include "AABB.h"
#include "glmHelp.h"
#include "Sphere.h"

AABB& AABB::operator=(const AABB& other) noexcept
{
	this->center = other.center;
	this->halfs = other.halfs;
	return *this;
}

Model AABB::GetModel() const
{
	// Will return the transform matrix for the unit cube centered at the origin with 2x2x2 dimension
	glm::vec3 transform = this->GetCenter();
	glm::vec3 scale     = this->Deviation();
	return Model(transform, glm::vec3(0), scale);
}


bool AABB::Overlap(const Sphere& other, Collision& collision) const
{
	glm::vec3 closest = glm::max(this->center - this->halfs, glm::min(other.center, this->center + this->halfs));
	float distance = glm::length(closest - other.center);
	collision.normal = glm::normalize(other.center - closest); // Get vector in the direction of the center from "here"
	if (glm::any(glm::isnan(collision.normal)))
	{
		collision.normal = closest;
	}
	collision.distance = other.radius - distance;
	collision.point = other.center + collision.normal * collision.distance;
	return distance < other.radius;
}