#pragma once
#ifndef SPHERE_H
#define SPHERE_H
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <tuple>
#include "AABB.h"
#include "Buffer.h"
#include "Vertex.h"
#include "Plane.h"
#include "CollisionTypes.h"

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

	inline bool RayCast(const Ray& ray) const noexcept;
	inline bool RayCast(const Ray& ray, RayCollision& out) const noexcept;

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

inline bool Sphere::RayCast(const Ray& ray, RayCollision& out) const noexcept
{
	out.Clear();

	bool result = glm::intersectRaySphere(ray.point, ray.direction, this->center, this->radius * this->radius, out.distance);
	if (result)
	{
		out.point = ray.point + ray.direction * out.distance;
		out.normal = glm::normalize(out.point - this->center);
	}

	return result;
	/*
	out.Clear();

	glm::vec3 delta = (this->center - ray.point);

	// Quadratic form
	float b = glm::dot(ray.direction, delta) * 2.f;
	float c = glm::length(delta) - this->radius * this->radius;

	float discriminant = b * b - 4 * c;

	// Miss
	if (discriminant < 0)
	{
		return false;
	}
	discriminant = glm::sqrt(discriminant);
	float near = (-b - discriminant) * 0.5f;
	float far  = (-b + discriminant) * 0.5f;
	if (glm::sign(near) < 0 && glm::sign(far) < 0)
	{
		return false;
	}
	if (near > far)
	{
		std::swap(near, far);
	}
	if (near < 0)
	{
		// This cannot happen?
	}
	

	out.point = ray.point + near * ray.direction;
	out.normal = glm::normalize(out.point - this->center);
	return out.depth > 0.f;
	*/
}

#endif // SPHERE_H

