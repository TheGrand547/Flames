#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <glm/gtx/intersect.hpp>
#include <vector>
#include "AABB.h"
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Lines.h"
#include <array>

class Plane;

class Triangle
{
protected:
	// Winding order 0 -> 1 -> 2, as expected
	glm::mat3 vertices;
public:
	inline Triangle() noexcept;
	inline Triangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) noexcept;
	inline Triangle(const glm::mat3& points) noexcept;

	inline glm::mat3 GetPoints() const noexcept;
	inline std::vector<glm::vec3> GetPointVector() const noexcept;
	inline std::array<glm::vec3, 3> GetPointArray() const noexcept;

	inline glm::vec3 GetNormal() const noexcept;
	inline glm::vec3 GetCenter() const noexcept;
	inline AABB GetAABB() const noexcept;

	bool SplitByPlane(const Plane& plane) const;
	bool Collinear(const Plane& plane) const;

	// True -> plane splits this triangle, orientation should be ignored
	// False -> plane does NOT split this triangle, orientation holds the relevant direction respective to the plane
	bool SplitAndOrientation(const Plane& plane, float& orientation) const;

	Plane GetPlane() const;
	// Assuming that the triangle is not split by the plane(NAN), at which point this result is worthless, which side is it on, +/0/-
	float GetSpatialRelation(const Plane& plane) const;

	// Figure out if these should be inlined or not
	inline bool ContainsPoint(const glm::vec3& point) const noexcept;

	inline bool RayCast(const Ray& ray) const noexcept;
	inline bool RayCast(const Ray& ray, RayCollision& collision) const noexcept;

	std::vector<Triangle> Split(const Plane& plane, bool cullBack = false) const;

	std::vector<Triangle> GetTriangles() const noexcept;

	glm::vec3 ClosestPoint(glm::vec3 vec3) const noexcept;
};

inline Triangle::Triangle() noexcept : vertices(1.f)
{

}

inline Triangle::Triangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) noexcept : vertices(a, b, c)
{
}

inline Triangle::Triangle(const glm::mat3& points) noexcept : vertices(points)
{
}

inline glm::mat3 Triangle::GetPoints() const noexcept
{
	return this->vertices;
}

inline std::vector<glm::vec3> Triangle::GetPointVector() const noexcept
{
	return std::vector<glm::vec3>({ this->vertices[0], this->vertices[1], this->vertices[2] });
}

inline std::array<glm::vec3, 3> Triangle::GetPointArray() const noexcept
{
	return std::to_array({ this->vertices[0], this->vertices[1], this->vertices[2] });
}

inline glm::vec3 Triangle::GetNormal() const noexcept
{
	return glm::normalize(glm::cross(this->vertices[1] - this->vertices[0], this->vertices[2] - this->vertices[0]));
}

inline glm::vec3 Triangle::GetCenter() const noexcept
{
	return (this->vertices[0] + this->vertices[1] + this->vertices[2]) / 3.f;
}

inline AABB Triangle::GetAABB() const noexcept
{
	return AABB::MakeAABB(this->vertices[0], this->vertices[1], this->vertices[2]);
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
	glm::vec2 barycentric{};
	bool result = glm::intersectRayTriangle(ray.point, ray.direction, this->vertices[0], this->vertices[1], this->vertices[2], barycentric, collision.depth);
	if (result)
	{
		collision.normal = this->GetNormal();
		collision.point = ray.point + collision.depth * ray.direction;
	}
	return result;
}

#endif // TRIANGLE_H
