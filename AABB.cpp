#include "AABB.h"
#include "glmHelp.h"
#include "Sphere.h"

AABB& AABB::operator=(const AABB& other) noexcept
{
	this->negativeBound = other.negativeBound;
	this->positiveBound = other.positiveBound;
	return *this;
}

Model AABB::GetModel() const
{
	// Will return the transform matrix for the unit cube centered at the origin with 2x2x2 dimension
	glm::vec3 transform = this->GetCenter();
	glm::vec3 scale     = this->Deviation();
	return Model(transform, glm::vec3(0, 0, 0), scale);
}


bool AABB::Overlap(const Sphere& other, Collision& collision) const
{
	glm::vec3 closest = glm::max(this->negativeBound, glm::min(other.center, this->positiveBound));
	float distance = glm::length(closest - other.center);

	collision.normal = glm::normalize(closest - other.center);
	collision.distance = other.radius - distance;
	collision.point = other.center + collision.normal * collision.distance;

	return distance < other.radius;
}