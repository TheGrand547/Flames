#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <vector>
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Lines.h"

class Plane;

class Triangle
{
protected:
	// Winding order 0 -> 1 -> 2, as expected
	glm::mat3 vertices;
	// TODO: Actually write this
	glm::vec3 normal;
public:
	constexpr Triangle();
	constexpr Triangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
	constexpr Triangle(const glm::mat3& points);

	constexpr glm::mat3 GetPoints() const noexcept;

	// Figure out if these should be inlined or not
	inline bool ContainsPoint(const glm::vec3& point) const noexcept;
	inline bool RayCast(const Ray& ray) const noexcept;
	inline bool RayCast(const Ray& ray, RayCollision& collision) const noexcept;

	std::vector<Triangle> Split(const Plane& plane) const;
};

constexpr Triangle::Triangle() : vertices(1.f), normal()
{

}

constexpr Triangle::Triangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) : vertices(a, b, c), normal()
{
}

constexpr Triangle::Triangle(const glm::mat3& points) : vertices(points), normal()
{
}

constexpr glm::mat3 Triangle::GetPoints() const noexcept
{
	return this->vertices;
}

// Implemented based on https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates/23745#23745
inline bool Triangle::ContainsPoint(const glm::vec3& point) const noexcept
{
	glm::vec3 edgeA = this->vertices[1] - this->vertices[0], edgeB = this->vertices[2] - this->vertices[0];
	glm::vec3 edgeC = point - this->vertices[0];
	float dotAA = glm::dot(edgeA, edgeA), dotAB = glm::dot(edgeA, edgeB), dotAC = glm::dot(edgeA, edgeC);
	float dotBB = glm::dot(edgeB, edgeB), dotBC = glm::dot(edgeB, edgeC);

	float inverse = 1.f / (dotAA * dotBB - dotAB * dotAB);
	float u = (dotBB * dotAC - dotAB * dotBC) * inverse;
	float v = (dotAA * dotBC - dotAB * dotAC) * inverse;

	// v2 * v0 =  u (v0 * v0) + v (v1 * v0)
	// v2 * v1 =  u (v0 * v1) + v (v1 * v1)

	// [v2*v0] = [v0*v0  v1*v0] [u]
	// [v2*v1] = [v0*v1  v1*v1] [v]

	// [v0*v0  v1*v0]^-1 [v2*v0] = [u]
	// [v0*v1  v1*v1]    [v2*v1] = [v]

	/* No clue what I was on about, think I was trying to re-derive it myself
	Vertex at ABC, point at P. Edges exist A->B(eB), A->C(eC) and A->P(eP) 
	[ePx] = u (eBx) + v (eCx)
	[ePy] = u (eBy) + v (eCy)
	[ePz] = u (eBz) + v (eCz)
	3x1  = 3x2 * 2x1
	*/

	return (u >= 0) && (v >= 0) && (u + v < 1);
}

inline bool Triangle::RayCast(const Ray& ray) const noexcept
{
	RayCollision dummy{};
	return this->RayCast(ray, dummy);
}

inline bool Triangle::RayCast(const Ray& ray, RayCollision& collision) const noexcept
{
	glm::vec3 edgeA = this->vertices[1] - vertices[0], edgeB = this->vertices[2] - vertices[1];
	glm::vec3 dirCrossB = glm::normalize(glm::cross(ray.direction, edgeB));

	float depth = glm::dot(dirCrossB, edgeA);

	// Parallel
	if (glm::abs(depth) < EPSILON)
	{
		return false; 
	}
	depth = 1.f / depth; // Invert it
	glm::vec3 deltaA = ray.point - this->vertices[0];
	// What does u even mean??
	float u = depth * glm::dot(deltaA, dirCrossB);
	// Dunno what
	if (u < 0 || u > 1)
	{
		return false;
	}

	glm::vec3 dirCrossA = glm::normalize(glm::cross(deltaA, edgeA));
	float v = depth * glm::dot(dirCrossA, ray.direction);
	if (v < 0 || u + v > 1)
	{
		return false;
	}
	float t = depth * glm::dot(edgeB, dirCrossA);
	if (t > EPSILON)
	{
		collision.axis = this->normal;
		collision.depth = t;
		collision.point = ray.point + t * ray.direction;
		return true;
	}
	else
	{
		// hit exists, but negative
	}
	return false;
}

#endif // TRIANGLE_H
