#pragma once
#ifndef PLANE_H
#define PLANE_H
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>
// glm::gtc::ext::intersection


// TODO: move to doubles?
class Plane
{
private:
	// ax + by + cx = d
	float constant;
	glm::vec3 normal;
	glm::vec3 point;
public:
	Plane(float a, float b, float c, float d) noexcept;
	Plane(const glm::vec3& vector, float f) noexcept;
	Plane(const glm::vec3& normal, const glm::vec3& point) noexcept;

	~Plane() noexcept = default;
	Plane(Plane&& other) noexcept = default;

	inline float Facing(const glm::vec3& vector) const noexcept;
	inline bool Intersects(const glm::vec3& pointA, const glm::vec3& pointB) const noexcept;
	inline bool IntersectsNormal(const glm::vec3& start, const glm::vec3& end) const noexcept;
	inline glm::vec3 PointOfIntersection(const glm::vec3& direction, const glm::vec3& point) const;
};

inline Plane::Plane(float a, float b, float c, float d) noexcept : normal(glm::vec3(a, b, c)), constant(d), point()
{
	// Maybe deprecated due to not having a defined point
}

inline Plane::Plane(const glm::vec3& vector, float f) noexcept : normal(vector), constant(f), point()
{
	// Point should be a point in the Plane
}

inline Plane::Plane(const glm::vec3& normal, const glm::vec3& point) noexcept : normal(normal), point(point), constant(glm::dot(normal, point))
{

}

inline float Plane::Facing(const glm::vec3& vector) const noexcept
{
	return glm::dot(this->normal, vector) - this->constant;
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

inline glm::vec3 Plane::PointOfIntersection(const glm::vec3& direction, const glm::vec3& point) const
{
	float dot = glm::dot(direction, this->normal);
	if (glm::abs(dot) < glm::epsilon<float>())
	{
		return glm::vec3(NAN, NAN, NAN);
	}
	float t = glm::dot(this->point - point, direction) / dot;
	return point + t * direction;
}
#endif