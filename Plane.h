#pragma once
#ifndef PLANE_H
#define PLANE_H
#include <glm/glm.hpp>
#include <glm/vector_relational.hpp>
#include <glm/gtx/vector_query.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "glmHelp.h"
#include "util.h"

// Set of points(P), defined by a normal(N) and a constant C such that P*N-C = 0
// Equivalently, for any given point A on the plane, (P-A)*N=0
class Plane
{
private:
	
	float constant;
	glm::vec3 normal;
public:
	Plane() noexcept;
	Plane(float a, float b, float c, float d) noexcept;
	Plane(const glm::vec3& vector, float f) noexcept;
	Plane(const glm::vec3& normal, const glm::vec3& point) noexcept;

	~Plane() noexcept = default;
	Plane(const Plane& other) noexcept;
	Plane(Plane&& other) noexcept = default;

	inline float GetConstant() const noexcept;
	inline glm::vec3 GetNormal() const noexcept;

	inline Plane& operator=(const Plane& other) noexcept;

	glm::vec3 GetPoint() const noexcept;

	glm::vec3 GetClosestPoint(glm::vec3 point) const noexcept;

	inline float Facing(const glm::vec3& vector) const noexcept;
	inline glm::vec3 Facing(const glm::mat3& points) const noexcept;
	inline float FacingNormal(const glm::vec3& vector) const noexcept;
	inline bool Intersects(const glm::vec3& pointA, const glm::vec3& pointB) const noexcept;
	inline bool IntersectsNormal(const glm::vec3& start, const glm::vec3& end) const noexcept;
	inline glm::vec3 PointOfIntersection(const glm::vec3& point, const glm::vec3& direction) const;

	bool TripleIntersect(const Plane& planeA, const Plane& planeB) const noexcept;
	bool TripleIntersect(const Plane& planeA, const Plane& planeB, glm::vec3& result) const noexcept;

	static inline bool TripleIntersect(const Plane& planeA, const Plane& planeB, const Plane& planeC);
	static inline bool TripleIntersect(const Plane& planeA, const Plane& planeB, const Plane& planeC, glm::vec3& result);
};

inline Plane::Plane() noexcept : normal(World::Forward), constant(0.f)
{
}

inline Plane::Plane(float a, float b, float c, float d) noexcept : normal(glm::normalize(glm::vec3(a, b, c))),
			constant(d / glm::length(glm::vec3(a, b, c)))
{
	assert(!glm::any(glm::isnan(this->normal)));
}

inline Plane::Plane(const glm::vec3& vector, float f) noexcept : normal(glm::normalize(vector)), constant(f / glm::length(vector))
{
	assert(!glm::any(glm::isnan(this->normal)));
}

inline Plane::Plane(const glm::vec3& normal, const glm::vec3& point) noexcept : normal(glm::normalize(normal)), 
																			constant(glm::dot(normal, point))
{
	assert(!glm::any(glm::isnan(this->normal)));
}

inline float Plane::GetConstant() const noexcept
{
	return this->constant;
}


inline glm::vec3 Plane::GetNormal() const noexcept
{
	return this->normal;
}

inline Plane& Plane::operator=(const Plane& other) noexcept
{
	this->constant = other.constant;
	this->normal   = other.normal;
	return *this;
}

inline float Plane::Facing(const glm::vec3& vector) const noexcept
{
	float value = glm::dot(this->normal, vector) - this->constant;
	return (glm::abs(value) < EPSILON) ? 0 : value;
}

inline glm::vec3 Plane::Facing(const glm::mat3& points) const noexcept
{
	glm::vec3 values = (this->normal * points) - glm::vec3(this->constant);
	values *= glm::not_(glm::isCompNull(values, EPSILON));
	return values;
}

inline float Plane::FacingNormal(const glm::vec3& vector) const noexcept
{
	return glm::dot(this->normal, vector);
}

inline bool Plane::Intersects(const glm::vec3& pointA, const glm::vec3& pointB) const noexcept
{
	float left = this->Facing(pointA);
	float right = this->Facing(pointB);
	return (left > 0) != (right > 0); // Can't be on the same side 
}

inline bool Plane::IntersectsNormal(const glm::vec3& start, const glm::vec3& end) const noexcept
{
	float left = this->Facing(start);
	float right = this->Facing(end);
	return (left > 0) && (right <= 0); // Allow movement from out to inbounds
}

inline glm::vec3 Plane::PointOfIntersection(const glm::vec3& point, const glm::vec3& direction) const
{
	// TODO: glm::gtx::intersection
	glm::vec3 norm = glm::normalize(direction);
	// TODO: glm::normalizedot
	float dot = glm::dot(norm, this->normal);
	if (glm::abs(dot) < glm::epsilon<float>())
	{
		return glm::vec3(NAN);
	}
	float t = glm::dot(this->GetPoint() - point, this->normal) / dot;
	return point + t * norm;
}

inline bool Plane::TripleIntersect(const Plane& planeA, const Plane& planeB, const Plane& planeC)
{
	return planeA.TripleIntersect(planeA, planeB, planeC);
}

inline bool Plane::TripleIntersect(const Plane& planeA, const Plane& planeB, const Plane& planeC, glm::vec3& result)
{
	return planeA.TripleIntersect(planeA, planeB, planeC, result);
}

#endif