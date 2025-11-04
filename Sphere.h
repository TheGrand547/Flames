#pragma once
#ifndef SPHERE_H
#define SPHERE_H
#include <glew.h>
#include <glm/glm.hpp>
#include <tuple>
#include "AABB.h"
#include "Buffer.h"
#include "Vertex.h"
#include "Plane.h"

struct Sphere
{
	glm::vec3 center;
	float radius;
	inline Sphere(const float& radius = 1.f, const glm::vec3& center = glm::vec3(0)) noexcept;
	inline Sphere(const glm::vec3& center, const float& radius = 1.f) noexcept;

	inline void Translate(const glm::vec3& amount) noexcept;
	inline void Scale(const float& amount) noexcept;

	inline float SignedDistance(const glm::vec3& point) const noexcept;

	inline bool FrontOrCollide(const Plane& plane) const noexcept;

	glm::mat4 GetModelMatrix() const noexcept;
	glm::mat4 GetNormalMatrix() const noexcept;

	inline AABB GetAABB() const noexcept;

	static void GenerateNormals(ArrayBuffer& verts, ElementArray& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
	static void GenerateMesh(ArrayBuffer& verts, ElementArray& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
	static void Generate(ArrayBuffer& verts, ElementArray& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
	static void GenerateLines(ElementArray& indicies, const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) noexcept;
};

inline Sphere::Sphere(const float& radius, const glm::vec3& center) noexcept : center(center), radius(radius)
{

}

inline Sphere::Sphere(const glm::vec3& center, const float& radius) noexcept : center(center), radius(radius)
{

}

inline void Sphere::Translate(const glm::vec3& amount) noexcept
{
	this->center += amount;
}

inline void Sphere::Scale(const float& amount) noexcept
{
	this->radius *= amount;
}

inline float Sphere::SignedDistance(const glm::vec3& point) const noexcept
{
	return glm::length(point - this->center) - this->radius;
}

inline bool Sphere::FrontOrCollide(const Plane& plane) const noexcept
{
	return plane.Facing(this->center) > -this->radius;
}

inline AABB Sphere::GetAABB() const noexcept
{
	return AABB(this->center - glm::vec3(this->radius), this->center + glm::vec3(this->radius));
}

#endif // SPHERE_H

