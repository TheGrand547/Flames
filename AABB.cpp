#include "AABB.h"
#include "glmHelp.h"

AABB& AABB::operator=(const AABB& other) noexcept
{
	this->negativeBound = other.negativeBound;
	this->positiveBound = other.positiveBound;
	return *this;
}

Model AABB::GetModel() const
{
	// Will return the transform matrix for the unit cube centered at the origin with 2x2x2 dimension
	glm::vec3 transform = (this->negativeBound + this->positiveBound) / 2.f;
	glm::vec3 scale     = (this->positiveBound - this->negativeBound) / 2.f;
	return Model(transform, glm::vec3(0, 0, 0), scale);
}