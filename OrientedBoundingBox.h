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
#include "Collidable.h"
#include "glmHelp.h"
#include "Plane.h"
#include "Sphere.h"
#include "util.h"

class OrientedBoundingBox
{
private:
	// TODO: Worth investigating if a 3x3 matrix would suffice
	glm::mat4 matrix;
	glm::vec3 halfs;
public:
	OrientedBoundingBox(const glm::vec3& euler = glm::vec3(0, 0, 0), const glm::vec3& deltas = glm::vec3(1, 1, 1));
	constexpr OrientedBoundingBox(const OrientedBoundingBox& other) = default;
	OrientedBoundingBox(const Model& model);
	constexpr OrientedBoundingBox(const AABB& other);
	~OrientedBoundingBox() = default;

	OrientedBoundingBox& operator=(const OrientedBoundingBox& other) = default;

	inline constexpr AABB GetAABB() const noexcept;

	inline constexpr glm::mat4 GetModelMatrix() const noexcept;
	inline constexpr glm::mat4 GetNormalMatrix() const noexcept;

	inline glm::vec3 Forward() const noexcept;
	inline glm::vec3 Up() const noexcept;
	inline glm::vec3 Cross() const noexcept;
	inline glm::vec3 operator[](const std::size_t& t) const;
	inline glm::vec3 Center() const noexcept;

	// TODO: Rethink the rotate/reorient from mat4 thing, replace with "Apply Transform" 
	inline void ReCenter(const glm::vec3& center) noexcept;

	inline void ReOrient(const glm::mat4& rotation);
	inline void ReOrient(const glm::vec3& euler);

	inline void ReScale(const glm::vec3& scale) noexcept;

	inline void Rotate(const glm::mat4& rotation);
	inline void Rotate(const glm::vec3& euler);

	inline void RotateAbout(const glm::mat4& rotation, const glm::vec3& point);
	inline void RotateAbout(const glm::vec3& euler, const glm::vec3& point);

	inline constexpr void Scale(const glm::vec3& scale);
	inline void Translate(const glm::vec3& distance) noexcept;

	// TODO: Move these to the classes in lines.h
	// TODO: Figure out why these are all inline
	// TODO: Pick a lane, constexpr or not
	// Don't do any of the extra math beyond determining if an intersection occurs
	inline constexpr bool FastIntersect(const glm::vec3& start, const glm::vec3& dir) const;

	inline constexpr bool Intersect(const glm::vec3& origin, const glm::vec3& dir) const;
	
	// If no intersection is found, distance is undefined
	inline constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const;
	
	// If no intersection is found, result is undefined
	inline constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& result) const;
	
	// If no intersection is found, near and far hit are undefined
	inline constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const;
	
	constexpr bool Overlap(const OrientedBoundingBox& other) const;
	constexpr bool Overlap(const OrientedBoundingBox& other, Collision& result) const;
	
	inline bool OverlapWithResponse(const OrientedBoundingBox& other);

	inline bool Overlap(const Sphere& other) const;
	bool Overlap(const Sphere& other, Collision& collision) const;

	// TODO: Constexpr?
	bool Overlap(const Capsule& other) const;
	bool Overlap(const Capsule& other, Collision& collision) const;

	// TODO: constexpr
	inline bool IntersectionWithResponse(const Plane& plane);
	inline bool Intersection(const Plane& plane) const;
	inline bool Intersection(const Plane& plane, float& distance) const;
	inline bool Intersection(const Plane& plane, Collision& out) const;

	glm::vec3 WorldToLocal(const glm::vec3& in) const;

	std::vector<glm::vec3> ClosestFacePoints(const glm::vec3& points) const;

	inline Model GetModel() const;
};

constexpr OrientedBoundingBox::OrientedBoundingBox(const AABB& other) : matrix(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), 
																				glm::vec4(other.GetCenter(), 1)), halfs(other.Deviation())
{

}

inline constexpr AABB OrientedBoundingBox::GetAABB() const noexcept
{
	glm::vec3 deviation(0.f);
	for (glm::length_t i = 0; i < 3; i++)
		deviation += glm::vec3(glm::abs(this->matrix[i])) * this->halfs[i];
	return AABB(glm::vec3(this->matrix[3]) - deviation, glm::vec3(this->matrix[3]) + deviation);
}

inline constexpr glm::mat4 OrientedBoundingBox::GetModelMatrix() const noexcept
{
	glm::mat4 model = this->GetNormalMatrix();
	for (glm::length_t i = 0; i < 3; i++)
		model[i] *= this->halfs[i];
	return model;
}

inline constexpr glm::mat4 OrientedBoundingBox::GetNormalMatrix() const noexcept
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

inline glm::vec3 OrientedBoundingBox::operator[](const std::size_t& t) const
{
	assert(t < 3);
	return this->matrix[(glm::length_t) t];
}

inline glm::vec3 OrientedBoundingBox::Center() const noexcept
{
	return this->matrix[3];
}

inline Model OrientedBoundingBox::GetModel() const
{
	glm::vec3 angles{ 0.f, 0.f, 0.f };
	glm::extractEulerAngleXYZ(this->matrix, angles.x, angles.y, angles.z);
	return Model(this->matrix[3], glm::degrees(angles), this->halfs);
}

inline void OrientedBoundingBox::ReCenter(const glm::vec3& center) noexcept
{
	this->matrix[3] = glm::vec4(center, 1);
}

inline void OrientedBoundingBox::ReOrient(const glm::vec3& euler)
{
	// TODO: Standardize using degrees or radians
	glm::vec4 center = this->matrix[3];
	this->matrix = glm::mat4(1.f);
	this->matrix[3] = center;
	this->Rotate(euler);
}

inline void OrientedBoundingBox::ReScale(const glm::vec3& scale) noexcept
{
	this->halfs = scale;
}

inline void OrientedBoundingBox::ReOrient(const glm::mat4& rotation)
{
	glm::vec4 center = this->matrix[3];
	this->matrix = glm::mat4(1.f);
	this->matrix[3] = center;
	this->Rotate(rotation);
}

inline void OrientedBoundingBox::Rotate(const glm::mat4& rotation)
{
	this->matrix *= rotation;
}

inline void OrientedBoundingBox::Rotate(const glm::vec3& euler)
{
	this->Rotate(glm::eulerAngleXYZ(glm::radians(euler.x), glm::radians(euler.y), glm::radians(euler.z)));
}

inline void OrientedBoundingBox::RotateAbout(const glm::mat4& rotation, const glm::vec3& point)
{
	// TODO: Something feels wrong about this
	//this->matrix = ((this->matrix * glm::translate(glm::mat4(1), point)) * rotation) * glm::translate(glm::mat4(1), -point);
	//this->matrix = glm::translate(glm::mat4(1), point) * rotation * glm::translate(glm::mat4(1), -point) * this->matrix;
	this->matrix = glm::translate(glm::mat4(1), point) * rotation * glm::translate(glm::mat4(1), -point) * this->matrix;
}

inline void OrientedBoundingBox::RotateAbout(const glm::vec3& euler, const glm::vec3& point)
{
	this->RotateAbout(glm::eulerAngleXYZ(glm::radians(euler.x), glm::radians(euler.y), glm::radians(euler.z)), point);
}

inline constexpr void OrientedBoundingBox::Scale(const glm::vec3& scale)
{
	this->halfs *= scale;
}

inline void OrientedBoundingBox::Translate(const glm::vec3& distance) noexcept
{
	this->matrix[3] += glm::vec4(distance, 0);
}

inline constexpr bool OrientedBoundingBox::FastIntersect(const glm::vec3& point, const glm::vec3& dir) const
{
	glm::vec3 delta = glm::vec3(this->Center()) - point;
	float nearHit = -std::numeric_limits<float>::infinity(), farHit = std::numeric_limits<float>::infinity();
	for (auto i = 0; i < 3; i++)
	{
		glm::vec3 axis = this->matrix[i];
		float scale = this->halfs[i];
		float parallel = glm::dot(axis, delta);
		if (glm::abs(glm::dot(dir, axis)) < EPSILON)
		{
			if (-parallel - scale > 0 || -parallel + scale > 0)
			{
				return false;
			}
		}

		float scaling = glm::dot(axis, dir);
		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit)
		{
			nearHit = param0;
		}
		if (param1 < farHit)
		{
			farHit = param1;
		}
		if (nearHit > farHit)
		{
			return false;
		}
		if (farHit < 0)
		{
			return false;
		}
	}
	return true;
}

inline constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir) const
{
	float dist;
	return this->Intersect(point, dir, dist);
}

inline constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const
{
	Collision collision{};
	bool value = this->Intersect(point, dir, collision);
	distance = collision.distance;
	return value;
}

inline constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& first) const
{
	Collision second;
	return this->Intersect(point, dir, first, second);
}

// https://www.sciencedirect.com/topics/computer-science/oriented-bounding-box
constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const
{
	nearHit.Clear();
	farHit.Clear();
	nearHit.distance = -std::numeric_limits<float>::infinity();
	farHit.distance = std::numeric_limits<float>::infinity();

	glm::vec3 delta = glm::vec3(this->Center()) - point;

	for (auto i = 0; i < 3; i++)
	{
		glm::vec3 axis = this->matrix[i];
		float scale = this->halfs[i];
		float parallel = glm::dot(axis, delta);
		float scaling = glm::dot(axis, dir);
		if (glm::abs(scaling) < EPSILON)
		{
			if (-parallel - scale > 0 || -parallel + scale > 0)
			{
				return false;
			}
		}

		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit.distance)
		{
			nearHit.distance = param0;
			nearHit.normal = axis * glm::sign(-parallel);
		}
		if (param1 < farHit.distance)
		{
			farHit.distance = param1;
			farHit.normal = axis * glm::sign(parallel);
		}
		if (nearHit.distance > farHit.distance)
		{
			return false;
		}
		if (farHit.distance < 0)
		{
			return false;
		}
	}
	nearHit.point = nearHit.distance * dir + point;
	farHit.point = farHit.distance * dir + point;
	if (nearHit.distance < 0)
	{
		std::swap(nearHit, farHit);
	}
	return true;
}

// https://web.stanford.edu/class/cs273/refs/obb.pdf
constexpr bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other) const
{
	Collision collide{};
	return this->Overlap(other, collide);
}

constexpr bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other, Collision& result) const
{
	std::array<glm::vec3, 15> separatingAxes{};
	for (glm::length_t i = 0; i < 3; i++)
	{
		separatingAxes[(std::size_t) i * 5] = this->matrix[i];
		separatingAxes[(std::size_t) i * 5 + 1] = other.matrix[i];
		for (glm::length_t j = 0; j < 3; j++)
		{
			separatingAxes[(std::size_t) i * 5 + 2 + j] = glm::normalize(glm::cross(glm::vec3(this->matrix[i]), glm::vec3(other.matrix[j])));
		}
	}
	glm::vec3 delta = this->Center() - other.Center();
	result.distance = INFINITY;
	glm::length_t index = 0;
	for (glm::length_t i = 0; i < separatingAxes.size(); i++)
	{
		glm::vec3 axis = separatingAxes[i];
		float left = glm::abs(glm::dot(axis, delta));
		float right = 0;

		for (glm::length_t i = 0; i < 3; i++)
		{
			right += glm::abs(this->halfs[i] * glm::dot(glm::vec3(this->matrix[i]), axis));
			right += glm::abs(other.halfs[i] * glm::dot(glm::vec3(other.matrix[i]), axis));
		}
		// This axis is a separating one 
		if (left > right)
		{
			return false;
		}
		if (result.distance > right - left)
		{
			index = i;
			result.distance = right - left;
		}
	}
	glm::vec3 normdir = glm::normalize(-delta); // direction here -> there
	// This is where *my* center needs to go to be not intersecting
	result.point = this->Center() + result.distance * separatingAxes[index] * glm::sign(-glm::dot(normdir, separatingAxes[index]));
	result.normal = separatingAxes[index] * glm::sign(-glm::dot(normdir, separatingAxes[index]));
	return true;
}



// TODO: This isn't inline or constexpr i'm just too lazy to move it
// TODO: Allow response to be applied later, and in different amounts so you can have real physics tm(ie small object moves/rotates more than big one etc)
inline bool OrientedBoundingBox::OverlapWithResponse(const OrientedBoundingBox& other)
{
	// SLOPPY
	if (this == &other) 
		return false;

	Collision collide;
	bool fool = this->Overlap(other, collide);
	if (fool)
	{
		glm::vec3 mineAxis{}, otherAxis{};
		//float myDot = INFINITY, otherDot = INFINITY;
		float myDot = -INFINITY, otherDot = -INFINITY;
		float distance = 0.f;
		
		bool tooAligned = false;
		for (glm::length_t i = 0; i < 3; i++)
		{
			float dotted = glm::abs(glm::dot(glm::vec3(this->matrix[i]), collide.normal));
			float dotted2 = glm::abs(glm::dot(glm::vec3(other.matrix[i]), collide.normal));
			if (1 - dotted < EPSILON)
				tooAligned = true;
			// Need signed projection on axis
			distance += glm::abs(this->halfs[i] * glm::dot(glm::vec3(this->matrix[i]), collide.normal));
			if (myDot < dotted)
			{
				mineAxis = this->matrix[i];
				myDot = dotted;
			}
			if (otherDot < dotted2)
			{
				otherAxis = other.matrix[i];
				otherDot = dotted2;
			}
		}

		// This is the point that will be rotated about, needs to be redone to solve the corner dilemma tm
		// TODO: CORNER DILEMMA
		glm::vec3 point = this->Center() + collide.normal * distance;
		
		glm::vec3 oldCenter = this->Center();

		this->matrix[3] = glm::vec4(collide.point, 1);
		// TODO: Still is too eager to turn along instead of towards the wall
		// TODO: Find closest corner and rotate towards it?
		// TODO: FIND THE CLOSEST ***EDGE*** AND ROTATE ALONG IT!!!!!! HOW???? I HAVE NO CLUE
		// Maybe do epsilon check?

		std::vector<glm::vec3> myPairs = this->ClosestFacePoints(other.Center()), otherPairs = other.ClosestFacePoints(this->Center());
		glm::vec3 myCenter = this->Center();
		glm::vec3 otherCenter = other.Center();
		float distanced = INFINITY;
		glm::vec3 bestRotate = glm::vec3(0);
		for (const glm::vec3& point : myPairs)
		{
			if (glm::length2(point - otherCenter) < distanced)
			{
				distanced = glm::length2(point - otherCenter);
				bestRotate = point;
			}
		}
		for (const glm::vec3& point : otherPairs)
		{
			if (glm::length2(point - myCenter) < distanced)
			{
				distanced = glm::length2(point - myCenter);
				bestRotate = point;
			}
		}
		collide.point = bestRotate;


		if (collide.distance > 0 && !tooAligned)
		{
			// Determine which side of the box the this point is, and rotate the "center" of it towards that
			//float direction = -glm::sign(glm::dot(collide.point, this->Cross()));
			float direction = -glm::sign(glm::dot(collide.point, this->Cross()) - glm::dot(oldCenter, this->Cross()));
			
			// Cross method needs to be paired with finding minimum of thing
			glm::vec3 cross = glm::cross(otherAxis, mineAxis);

			// TODO: maybe refire the collision detection to stop it from flickering?
			if (!glm::all(glm::lessThan(glm::abs(cross), glm::vec3(EPSILON))))
				this->RotateAbout(glm::rotate(glm::mat4(1.f), collide.distance * direction, cross), collide.point);
		}
	}
	return fool;
}

inline bool OrientedBoundingBox::Overlap(const Sphere& other) const
{
	Collision local;
	return this->Overlap(other, local);
}

inline bool OrientedBoundingBox::Intersection(const Plane& plane, Collision& collision) const
{
	float delta = plane.Facing(this->Center());
	collision.normal = plane.GetNormal();

	// Ensure that the box can always go from out to inbounds
	if (!plane.TwoSided() && (delta < 0 || delta > glm::length(this->halfs)))
		return false;

	float projected = 0.f;

	for (glm::length_t i = 0; i < 3; i++)
		projected += glm::abs(glm::dot(glm::vec3(this->matrix[i] * this->halfs[i]), plane.GetNormal()));

	collision.distance = projected - glm::abs(delta);
	collision.point = this->Center() + glm::sign(delta) * glm::abs(collision.distance) * collision.normal; // This might be wrong?
	return glm::abs(projected) > glm::abs(delta);
}

inline bool OrientedBoundingBox::Intersection(const Plane& plane, float& distance) const
{
	Collision collision{};
	bool result = this->Intersection(plane, collision);
	distance = collision.distance;
	return result;
}

inline bool OrientedBoundingBox::Intersection(const Plane& plane) const
{
	Collision collision{};
	return this->Intersection(plane, collision);
}

inline bool OrientedBoundingBox::IntersectionWithResponse(const Plane& plane)
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
/*
// Triangles are laid out like (123) (234) (345) in the list, repeated tris
OBB MakeOBB(std::vector<glm::vec3> triangles)
{
	return OBB
}

OBB 
*/
#endif // ORIENTED_BOUNDING_BOX_H
