#include "AABB.h"

AABB& AABB::operator=(const AABB& other) noexcept
{
	this->negativeBound = other.negativeBound;
	this->positiveBound = other.positiveBound;
	return *this;
}
