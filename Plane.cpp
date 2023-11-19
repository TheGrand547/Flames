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

bool Plane::TripleIntersect(const Plane& planeA, const Plane& planeB) const noexcept
{
	glm::vec3 _dummy{};
	return this->TripleIntersect(planeA, planeB, _dummy);
}

bool Plane::TripleIntersect(const Plane& planeA, const Plane& planeB, glm::vec3& result) const noexcept
{
	glm::vec3 n0 = this->normal, n1 = planeA.normal, n2 = planeB.normal;
	float denom = glm::dot(glm::cross(n0, n1), n2);
	if (glm::abs(denom) > EPSILON)
	{
		result = glm::cross(n1, n2) * this->constant;
		result += glm::cross(n2, n0) * planeA.constant;
		result += glm::cross(n0, n1) * planeB.constant;
		result /= denom;
		return true;
	}
	return false;
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