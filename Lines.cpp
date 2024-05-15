#include "Lines.h"
#include <array>
#include "Plane.h"

// ** LINEBASE ** \\


float LineBase::Distance(const LineBase& other) const noexcept
{
	glm::vec3 a, b;
	return this->Distance(other, a, b);
}

float LineBase::Distance(const LineBase& other, glm::vec3& thisPoint, glm::vec3& otherPoint) const noexcept
{
	std::array<glm::vec3, 4> dirs =
	{
		other.PointA() - this->PointA(),
		other.PointA() - this->PointB(),
		other.PointB() - this->PointA(),
		other.PointB() - this->PointB()
	};

	if (glm::length2(dirs[1]) < glm::length2(dirs[0]) ||
		glm::length2(dirs[1]) < glm::length2(dirs[2]) ||
		glm::length2(dirs[3]) < glm::length2(dirs[0]) ||
		glm::length2(dirs[3]) < glm::length2(dirs[2]))
	{
		thisPoint = this->PointB();
	}
	else
	{
		thisPoint = this->PointA();
	}
	otherPoint = other.PointClosestTo(thisPoint);
	thisPoint = this->PointClosestTo(otherPoint);

	return glm::length(otherPoint - thisPoint);
}

// ** LINE ** \\

glm::vec3 Line::PointA() const noexcept
{
	return this->initial;
}

glm::vec3 Line::PointB() const noexcept
{
	return this->initial + dir;
}

// ** RAY ** \\

glm::vec3 Ray::PointClosestTo(const glm::vec3& point) const noexcept
{
	return this->point + glm::max(glm::dot(this->point - point, this->dir), 0.f) * this->dir;
}

// ** LINE SEGMENT **\\

// https://stackoverflow.com/a/18994296
float LineSegment::Length() const noexcept
{
	return glm::length(this->A - this->B);
}

float LineSegment::Magnitude() const noexcept
{
	return this->Length();
}

float LineSegment::SquaredLength() const noexcept
{
	return glm::length2(this->A - this->B);
}

glm::vec3 Line::PointClosestTo(const glm::vec3& point) const noexcept
{
	return this->point + glm::dot(this->point - point, this->dir) * this->dir;
}

glm::vec3 LineSegment::PointClosestTo(const glm::vec3& point) const noexcept
{
	glm::vec3 delta = this->B - this->A;
	float along = glm::dot(point - this->A, delta) / glm::length2(delta);
	return this->A + glm::min(glm::max(along, 0.f), 1.f) * delta;
}

std::vector<LineSegment> LineSegment::Split(const Plane& plane) const
{
	std::vector<LineSegment> spliced;
	float aPlace = plane.Facing(this->A), bPlace = plane.Facing(this->B);
	if (glm::sign(aPlace) == glm::sign(bPlace) || (glm::sign(aPlace) == 0.f || glm::sign(bPlace) == 0.f))
	{
		spliced.push_back(*this);
	}
	else
	{
		glm::vec3 point = plane.PointOfIntersection(this->A, this->UnitDirection());
		spliced.emplace_back(this->A, point);
		spliced.emplace_back(this->B, point);
	}

	return spliced;
}