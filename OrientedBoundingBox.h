#pragma once
#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H
#include <array>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <glm/glm.hpp>
#include <limits>
#include "AABB.h"
#include "Capsule.h"
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Plane.h"
#include "Sphere.h"
#include "Triangle.h"
#include "util.h"

class OrientedBoundingBox
{
private:
	// TODO: Worth investigating if a 3x3 matrix would suffice
	glm::mat4 matrix;
	glm::vec3 halfs;
public:
	OrientedBoundingBox(const glm::vec3& euler = glm::vec3(0, 0, 0), const glm::vec3& deltas = glm::vec3(1, 1, 1)) noexcept;
	inline OrientedBoundingBox(const OrientedBoundingBox& other) noexcept = default;
	OrientedBoundingBox(const Model& model) noexcept;
	inline OrientedBoundingBox(const AABB& other) noexcept;
	~OrientedBoundingBox() noexcept = default;

	OrientedBoundingBox& operator=(const OrientedBoundingBox& other) noexcept = default;

	inline AABB GetAABB() const noexcept;

	inline glm::mat4 GetModelMatrix() const noexcept;
	inline glm::mat4 GetNormalMatrix() const noexcept;

	inline glm::vec3 Forward() const noexcept;
	inline glm::vec3 Up() const noexcept;
	inline glm::vec3 Cross() const noexcept;
	inline glm::vec3 operator[](const std::size_t& t) const noexcept;
	inline glm::vec3 Center() const noexcept;

	inline glm::vec3 GetScale() const noexcept;

	inline float SignedDistance(const glm::vec3& point) const noexcept;

	// TODO: Rethink the rotate/reorient from mat4 thing, replace with "Apply Transform" 
	inline void ReCenter(const glm::vec3& center) noexcept;

	inline void ReOrient(const glm::mat4& rotation) noexcept;

	// Takes input in degrees
	inline void ReOrient(const glm::vec3& euler) noexcept;

	inline void ReScale(const glm::vec3& scale) noexcept;

	inline void Rotate(const glm::mat4& rotation) noexcept;

	// Takes input in degrees
	inline void Rotate(const glm::vec3& euler) noexcept;

	inline void RotateAbout(const glm::mat4& rotation, const glm::vec3& point) noexcept;
	inline void RotateAbout(const glm::vec3& euler, const glm::vec3& point) noexcept;

	inline void Scale(const glm::vec3& scale) noexcept;
	inline void Scale(const float& scale) noexcept;
	inline void Translate(const glm::vec3& distance) noexcept;

	// TODO: Move these to the classes in lines.h

	// Don't do any of the extra math beyond determining if an intersection occurs
	bool FastIntersect(const glm::vec3& start, const glm::vec3& dir) const noexcept;

	inline bool Intersect(const glm::vec3& origin, const glm::vec3& dir) const noexcept;
	
	// If no intersection is found, distance is undefined
	inline bool Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const noexcept;
	
	// If no intersection is found, result is undefined
	inline bool Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& result) const noexcept;
	
	// If no intersection is found, near and far hit are undefined
	bool Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& nearHit, RayCollision& farHit) const noexcept;
	
	inline bool Overlap(const OrientedBoundingBox& other) const noexcept;
	bool Overlap(const OrientedBoundingBox& other, SlidingCollision& result) const noexcept;
	bool Overlap(const OrientedBoundingBox& other, SlidingCollision& slide, RotationCollision& rotate) const noexcept;
	
	// These both assume 'this' is dynamic, and the other is static, other methods will handle the case of both being dynamic
	inline bool OverlapAndSlide(const OrientedBoundingBox& other) noexcept;
	bool OverlapCompleteResponse(const OrientedBoundingBox& other) noexcept;


	inline bool Overlap(const Sphere& other) const noexcept;
	bool Overlap(const Sphere& other, Collision& collision) const noexcept;

	inline bool Overlap(const Capsule& other) const noexcept;
	bool Overlap(const Capsule& other, Collision& collision) const noexcept;

	bool Intersection(const Plane& plane) const noexcept;
	inline bool Intersection(const Plane& plane, float& distance) const noexcept;
	inline bool Intersection(const Plane& plane, Collision& out) const noexcept;
	inline bool IntersectionWithResponse(const Plane& plane) noexcept;

	inline float ProjectionLength(const glm::vec3& vector) const noexcept;

	glm::vec3 WorldToLocal(const glm::vec3& in) const noexcept;

	std::array<LineSegment, 12> GetLineSegments() const noexcept;
	std::vector<Triangle> GetTriangles() const noexcept;

	inline Model GetModel() const noexcept;

	// Trust the user to not do this erroneously
	inline void ApplyCollision(const SlidingCollision& collision) noexcept;
	inline void ApplyCollision(const RotationCollision& collision) noexcept;
};

inline OrientedBoundingBox::OrientedBoundingBox(const AABB& other) noexcept : matrix(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0),
																				glm::vec4(other.GetCenter(), 1)), halfs(other.Deviation())
{

}

inline AABB OrientedBoundingBox::GetAABB() const noexcept
{
	glm::vec3 deviation(0.f);
	for (glm::length_t i = 0; i < 3; i++)
		deviation += glm::vec3(glm::abs(this->matrix[i])) * this->halfs[i];
	return AABB(glm::vec3(this->matrix[3]) - deviation, glm::vec3(this->matrix[3]) + deviation);
}

inline glm::mat4 OrientedBoundingBox::GetModelMatrix() const noexcept
{
	glm::mat4 model = this->GetNormalMatrix();
	for (glm::length_t i = 0; i < 3; i++)
		model[i] *= this->halfs[i];
	return model;
}

inline glm::mat4 OrientedBoundingBox::GetNormalMatrix() const noexcept
{
	return this->matrix;
}

inline glm::vec3 OrientedBoundingBox::Forward() const noexcept
{
	return this->matrix[0];
}

inline glm::vec3 OrientedBoundingBox::Up() const noexcept
{
	return this->matrix[1];
}

inline glm::vec3 OrientedBoundingBox::Cross() const noexcept
{
	return this->matrix[2];
}

inline glm::vec3 OrientedBoundingBox::operator[](const std::size_t& t) const noexcept
{
	assert(t < 3);
	return this->matrix[static_cast<glm::length_t>(t)];
}

inline glm::vec3 OrientedBoundingBox::Center() const noexcept
{
	return this->matrix[3];
}

inline glm::vec3 OrientedBoundingBox::GetScale() const noexcept
{
	return this->halfs;
}

inline float OrientedBoundingBox::SignedDistance(const glm::vec3& point) const noexcept
{
	glm::vec3 transformed = glm::abs(this->WorldToLocal(point)) - this->halfs;
	return glm::length(glm::max(transformed, glm::vec3(0.f))) + glm::min(glm::compMax(transformed), 0.f);
}

inline Model OrientedBoundingBox::GetModel() const noexcept
{
	glm::vec3 angles{ 0.f, 0.f, 0.f };
	glm::extractEulerAngleXYZ(this->matrix, angles.x, angles.y, angles.z);
	return Model(this->matrix[3], glm::degrees(angles), this->halfs);
}

inline void OrientedBoundingBox::ApplyCollision(const SlidingCollision& collision) noexcept
{
	this->matrix[3] += glm::vec4(collision.normal * (collision.distance + 0), 0);
}

inline void OrientedBoundingBox::ApplyCollision(const RotationCollision& collision) noexcept
{
	if (glm::abs(collision.distance) > EPSILON)
		this->RotateAbout(glm::rotate(glm::mat4(1.f), collision.distance, collision.axis), collision.point);
}

inline void OrientedBoundingBox::ReCenter(const glm::vec3& center) noexcept
{
	this->matrix[3] = glm::vec4(center, 1);
}

inline void OrientedBoundingBox::ReOrient(const glm::vec3& euler) noexcept
{
	glm::vec4 center = this->matrix[3];
	this->matrix = glm::mat4(1.f);
	this->Rotate(euler);
	this->matrix[3] = center;
}

inline void OrientedBoundingBox::ReScale(const glm::vec3& scale) noexcept
{
	this->halfs = scale;
}

inline void OrientedBoundingBox::ReOrient(const glm::mat4& rotation) noexcept
{
	glm::vec4 center = this->matrix[3];
	this->matrix = glm::mat4(1.f);
	this->matrix[3] = center;
	this->Rotate(rotation);
}

inline void OrientedBoundingBox::Rotate(const glm::mat4& rotation) noexcept
{
	this->matrix *= rotation;
}

inline void OrientedBoundingBox::Rotate(const glm::vec3& euler) noexcept
{
	glm::vec3 temp(glm::radians(euler));
	this->Rotate(glm::eulerAngleXYZ(temp.x, temp.y, temp.z));
}

inline void OrientedBoundingBox::RotateAbout(const glm::mat4& rotation, const glm::vec3& point) noexcept
{
	this->matrix = glm::translate(glm::mat4(1), point) * rotation * glm::translate(glm::mat4(1), -point) * this->matrix;
}

inline void OrientedBoundingBox::RotateAbout(const glm::vec3& euler, const glm::vec3& point) noexcept
{
	glm::vec3 temp(glm::radians(euler));
	this->RotateAbout(glm::eulerAngleXYZ(temp.x, temp.y, temp.z), point);
}

inline void OrientedBoundingBox::Scale(const glm::vec3& scale) noexcept
{
	this->halfs *= scale;
}

inline void OrientedBoundingBox::Scale(const float& scale) noexcept
{
	this->halfs *= scale;
}

inline void OrientedBoundingBox::Translate(const glm::vec3& distance) noexcept
{
	this->matrix[3] += glm::vec4(distance, 0);
}

inline bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir) const noexcept
{
	float dist;
	return this->Intersect(point, dir, dist);
}

inline bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const noexcept
{
	RayCollision collision{};
	bool value = this->Intersect(point, dir, collision);
	distance = collision.distance;
	return value;
}

inline bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& first) const noexcept
{
	RayCollision second;
	return this->Intersect(point, dir, first, second);
}

inline bool OrientedBoundingBox::Overlap(const Capsule& other) const noexcept
{
	Collision collision{};
	return this->Overlap(other, collision);
}

inline bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other) const noexcept
{
	SlidingCollision collide{};
	return this->Overlap(other, collide);
}

inline bool OrientedBoundingBox::OverlapAndSlide(const OrientedBoundingBox& other) noexcept
{
	// SLOPPY
	if (this == &other) 
		return false;

	SlidingCollision collide;
	bool fool = this->Overlap(other, collide);
	if (fool)
	{
		this->ApplyCollision(collide);
	}
	return fool;
}

inline bool OrientedBoundingBox::Overlap(const Sphere& other) const noexcept
{
	Collision local;
	return this->Overlap(other, local);
}

inline float OrientedBoundingBox::ProjectionLength(const glm::vec3& vector) const noexcept
{
	float result = 0.f;
	result = glm::dot(glm::abs(vector * glm::mat3(this->matrix)), this->halfs);
	return result;
}

inline bool OrientedBoundingBox::Intersection(const Plane& plane, float& distance) const noexcept
{
	Collision collision{};
	bool result = this->Intersection(plane, collision);
	distance = collision.distance;
	return result;
}

inline bool OrientedBoundingBox::Intersection(const Plane& plane) const noexcept
{
	Collision collision{};
	return this->Intersection(plane, collision);
}

inline bool OrientedBoundingBox::IntersectionWithResponse(const Plane& plane) noexcept
{
	Collision collision{};
	bool result = this->Intersection(plane, collision);
	if (result)
	{
		this->matrix[3] = glm::vec4(collision.point, 1);
	}
	return result;
}

typedef OrientedBoundingBox OBB;
#endif // ORIENTED_BOUNDING_BOX_H
