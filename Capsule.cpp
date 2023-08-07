#include "Capsule.h"

bool Capsule::Intersect(const Capsule& other) const noexcept
{
	Collision temp;
	return this->Intersect(other, temp);
}

bool Capsule::Intersect(const Capsule& other, Collision& hit) const noexcept
{
	/*
	std::array<glm::vec3, 4> dirs =
	{
		other.line.A - this->line.A,
		other.line.A - this->line.B,
		other.line.B - this->line.A,
		other.line.B - this->line.B
	};
	glm::vec3 bestA, bestB;
	if (glm::length2(dirs[1]) < glm::length2(dirs[0]) ||
		glm::length2(dirs[1]) < glm::length2(dirs[2]) ||
		glm::length2(dirs[3]) < glm::length2(dirs[0]) ||
		glm::length2(dirs[3]) < glm::length2(dirs[2]))
	{
		bestA = this->line.B;
	}
	else
	{
		bestA = this->line.A;
	}
	bestB = other.line.PointClosestTo(bestA);
	bestA = this->line.PointClosestTo(bestB);
	*/
	glm::vec3 bestA(0), bestB(0);
	float distance = this->line.Distance(other.line, bestA, bestB);
	hit.normal = bestB - bestA;
	hit.normal = glm::normalize(hit.normal);
	hit.distance = this->radius + other.radius - distance;
	hit.point = bestB + other.radius * hit.normal;
	return distance > 0;
}