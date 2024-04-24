#pragma once
#ifndef PLANE_H
#define PLANE_H
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "glmHelp.h"
// glm::gtc::ext::intersection

// TODO: Get it together with with normalizd normal god dammit
class Plane
{
private:
	// ax + by + cx = d
	float constant;
	glm::vec3 normal;
	glm::vec3 point;
	bool twoSided;
	
	void CalculatePoint();

public:
	Plane(float a, float b, float c, float d, bool twoSided = false) noexcept;
	Plane(const glm::vec3& vector, float f, bool twoSided = false) noexcept;
	Plane(const glm::vec3& normal, const glm::vec3& point, bool twoSided = false) noexcept;

	~Plane() noexcept = default;
	Plane(const Plane& other) noexcept;
	Plane(Plane&& other) noexcept = default;

	inline bool TwoSided() const noexcept;
	inline float GetConstant() const noexcept;
	inline glm::vec3 GetNormal() const noexcept;
	inline void Normalize();
	inline void ToggleTwoSided() noexcept;

	inline Plane& operator=(const Plane& other) noexcept;

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
// TODO: reorder these dummy

inline Plane::Plane(float a, float b, float c, float d, bool twoSided) noexcept : normal(a, b, c), constant(d), 
																					point(), twoSided(twoSided)
{
	assert(normal != glm::vec3(0));
	this->Normalize();
	this->CalculatePoint();
}

inline Plane::Plane(const glm::vec3& vector, float f, bool twoSided) noexcept : normal(vector), constant(f), point(), twoSided(twoSided)
{
	assert(normal != glm::vec3(0));
	this->Normalize();
	this->CalculatePoint();
}

inline Plane::Plane(const glm::vec3& normal, const glm::vec3& point, bool twoSided) noexcept : normal(normal), point(point), 
																				constant(glm::dot(normal, point)), twoSided(twoSided)
{
	this->Normalize();
}

inline bool Plane::TwoSided() const noexcept
{
	return this->twoSided;
}

inline void Plane::Normalize()
{
	float length = glm::length(this->normal);
	this->normal /= length;
	this->constant /= length;
}

inline void Plane::ToggleTwoSided() noexcept
{
	this->twoSided = !this->twoSided;
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
	this->point    = other.point;
	this->twoSided = other.twoSided;
	return *this;
}

// TODO: Investigate zeroing out close to zero values with EPSILON
inline float Plane::Facing(const glm::vec3& vector) const noexcept
{
	return glm::dot(this->normal, vector) - this->constant;
}

inline glm::vec3 Plane::Facing(const glm::mat3& points) const noexcept
{	
	return this->normal * points - glm::vec3(this->constant);
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
	float dot = glm::dot(glm::normalize(direction), glm::normalize(this->normal));
	if (glm::abs(dot) < glm::epsilon<float>())
	{
		return glm::vec3(NAN, NAN, NAN);
	}
	float t = glm::dot(this->point - point, glm::normalize(this->normal)) / dot;
	return point + t * glm::normalize(direction);
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