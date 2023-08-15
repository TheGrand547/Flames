#include "Plane.h"
#include "util.h"
#include "log.h"

Plane::Plane(const Plane& other) noexcept
{
	this->constant = other.constant;
	this->normal   = other.normal;
	this->point    = other.point;
	this->twoSided = other.twoSided;
}

void Plane::CalculatePoint()
{
	glm::vec3 vec(0);
	if (glm::abs(this->normal.x) > EPSILON)
	{
		vec = glm::vec3(1, 0, 0);
		this->point = vec * this->constant / this->normal.x;
	}
	else if (glm::abs(this->normal.y) > EPSILON)
	{
		vec = glm::vec3(0, 1, 0);
		this->point = vec * this->constant / this->normal.y;
	}
	else if (glm::abs(this->normal.z) > EPSILON)
	{
		vec = glm::vec3(0, 0, 1);
		this->point = vec * this->constant / this->normal.z;
	}
	else
	{
		LogF("Plane created with invalid normal\n");
		return;
	}
	assert(glm::abs(this->Facing(this->point)) < EPSILON);
}