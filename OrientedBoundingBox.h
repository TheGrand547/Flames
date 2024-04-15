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

	inline glm::vec3 GetScale() const noexcept;

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
	inline constexpr void Scale(const float& scale);
	inline void Translate(const glm::vec3& distance) noexcept;

	// TODO: Move these to the classes in lines.h
	// TODO: Figure out why these are all inline
	// TODO: Pick a lane, constexpr or not

	// Don't do any of the extra math beyond determining if an intersection occurs
	constexpr bool FastIntersect(const glm::vec3& start, const glm::vec3& dir) const;

	inline constexpr bool Intersect(const glm::vec3& origin, const glm::vec3& dir) const;
	
	// If no intersection is found, distance is undefined
	inline constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const;
	
	// If no intersection is found, result is undefined
	inline constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& result) const;
	
	// If no intersection is found, near and far hit are undefined
	constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& nearHit, RayCollision& farHit) const;
	
	inline constexpr bool Overlap(const OrientedBoundingBox& other) const;
	constexpr bool Overlap(const OrientedBoundingBox& other, SlidingCollision& result) const;
	bool Overlap(const OrientedBoundingBox& other, SlidingCollision& slide, RotationCollision& rotate) const;
	
	// These both assume 'this' is dynamic, and the other is static, other methods will handle the case of both being dynamic
	inline bool OverlapAndSlide(const OrientedBoundingBox& other);
	bool OverlapCompleteResponse(const OrientedBoundingBox& other);


	inline bool Overlap(const Sphere& other) const;
	bool Overlap(const Sphere& other, Collision& collision) const;

	// TODO: Constexpr?
	bool Overlap(const Capsule& other) const;
	bool Overlap(const Capsule& other, Collision& collision) const;

	// TODO: constexpr
	bool Intersection(const Plane& plane) const;
	inline bool Intersection(const Plane& plane, float& distance) const;
	inline bool Intersection(const Plane& plane, Collision& out) const;
	inline bool IntersectionWithResponse(const Plane& plane);

	inline float ProjectionLength(const glm::vec3& vector) const;

	glm::vec3 WorldToLocal(const glm::vec3& in) const;

	std::array<LineSegment, 12> GetLineSegments() const;

	inline Model GetModel() const;

	// Trust the user to not do this erroneously
	inline constexpr void ApplyCollision(const SlidingCollision& collision);
	inline constexpr void ApplyCollision(const RotationCollision& collision);
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

inline glm::vec3 OrientedBoundingBox::GetScale() const noexcept
{
	return this->halfs;
}

inline Model OrientedBoundingBox::GetModel() const
{
	glm::vec3 angles{ 0.f, 0.f, 0.f };
	glm::extractEulerAngleXYZ(this->matrix, angles.x, angles.y, angles.z);
	return Model(this->matrix[3], glm::degrees(angles), this->halfs);
}

inline constexpr void OrientedBoundingBox::ApplyCollision(const SlidingCollision& collision)
{
	this->matrix[3] += glm::vec4(collision.normal * (collision.distance + 0), 0);
}

inline constexpr void OrientedBoundingBox::ApplyCollision(const RotationCollision& collision)
{
	if (glm::abs(collision.distance) > EPSILON)
		this->RotateAbout(glm::rotate(glm::mat4(1.f), collision.distance, collision.axis), collision.point);
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
	this->Rotate(euler);
	this->matrix[3] = center;
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

inline constexpr void OrientedBoundingBox::Scale(const float& scale)
{
	this->halfs *= scale;
}

inline void OrientedBoundingBox::Translate(const glm::vec3& distance) noexcept
{
	this->matrix[3] += glm::vec4(distance, 0);
}

constexpr bool OrientedBoundingBox::FastIntersect(const glm::vec3& point, const glm::vec3& dir) const
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
	RayCollision collision{};
	bool value = this->Intersect(point, dir, collision);
	distance = collision.distance;
	return value;
}

inline constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& first) const
{
	RayCollision second;
	return this->Intersect(point, dir, first, second);
}

// https://www.sciencedirect.com/topics/computer-science/oriented-bounding-box
constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, RayCollision& nearHit, RayCollision& farHit) const
{
	// TODO: For line segments do the clamp thingy
	nearHit.Clear();
	farHit.Clear();
	nearHit.distance = -std::numeric_limits<float>::infinity();
	farHit.distance = std::numeric_limits<float>::infinity();

	glm::vec3 delta = this->Center() - point;
	for (auto i = 0; i < 3; i++)
	{
		glm::vec3 axis = this->matrix[i];
		float scale = this->halfs[i];

		float parallel = glm::dot(axis, delta); // Distance from Point to my center, in the direction of this axis
		float scaling = glm::dot(axis, dir);    // Length of projection of dir onto this axis

		// Check if the direction is parallel to one of the faces
		if (glm::abs(scaling) < EPSILON)
		{
			if (abs(parallel) > scale)
			{
				//std::cout << "Parallel check" << std::endl;
				return false;
			}
			else
			{
				//continue;
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
			//1std::cout << "Inversion check" << std::endl;
			return false;
		}
		if (farHit.distance < 0)
		{
			//std::cout << "Farness test" << std::endl;
			return false;
		}
		//std::cout << std::endl;
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
inline constexpr bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other) const
{
	SlidingCollision collide{};
	return this->Overlap(other, collide);
}

constexpr bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other, SlidingCollision& result) const
{
	const glm::mat<3, 2, const glm::length_t> indexLookUp{ {2, 1}, {2, 0}, { 1, 0} };
	std::array<glm::vec3, 15> separatingAxes{};
	glm::mat3 dotProducts{};
	glm::mat3 crossLengths{};
	for (glm::length_t i = 0; i < 3; i++)
	{
		glm::vec3 myAxis = this->matrix[i];
		separatingAxes[static_cast<std::size_t>(i) * 5] = myAxis;
		separatingAxes[static_cast<std::size_t>(i) * 5 + 1] = other.matrix[i];
		for (glm::length_t j = 0; j < 3; j++)
		{
			glm::vec3 otherAxis = other.matrix[j];
			glm::vec3 crossResult = glm::cross(myAxis, glm::vec3(otherAxis));

			float crossLength = 1.f / glm::length(crossResult);
			separatingAxes[static_cast<std::size_t>(i) * 5 + 2 + j] = crossResult * crossLength;

			dotProducts[i][j] = glm::abs(glm::dot(myAxis, otherAxis));
			crossLengths[i][j] = crossLength;
		}
	}
	// If the least separating axis is one of the 6 face normals it's a corner-edge collision, otherwise it's edge-edge
	const glm::vec3 delta = this->Center() - other.Center();
	result.distance = INFINITY;
	glm::length_t index = 0;
	for (glm::length_t i = 0; i < separatingAxes.size(); i++)
	{
		glm::vec3 axis = separatingAxes[i];
		const float deltaProjection = glm::abs(glm::dot(axis, delta));
		float obbProjections = 0;
		glm::length_t truncatedIndex = i % 5;
		glm::length_t axialIndex = i / 5;

		if (truncatedIndex == 0) // Axis from this OBB
		{
			/*
			obbProjections = this->halfs[axialIndex] + other.halfs[0] * dotProducts[axialIndex][0] + other.halfs[1] * dotProducts[axialIndex][1]
				+ other.halfs[2] * dotProducts[axialIndex][2];
				*/
			// I think this is correct?
			obbProjections = this->halfs[axialIndex] + glm::dot(other.halfs, dotProducts[axialIndex]);
		}
		else if (truncatedIndex == 1) // Axis from the other OBB
		{
			obbProjections = other.halfs[axialIndex] + this->halfs[0] * dotProducts[0][axialIndex] + this->halfs[1] * dotProducts[1][axialIndex]
				+ this->halfs[2] * dotProducts[2][axialIndex];
		}
		else
		{	 
			// Truncated index is [2,4], map to [0, 2]
			// Axis is this[axialIndex] cross other[truncatedIndex - 2]
			truncatedIndex -= 2;

			glm::length_t firstPairA = indexLookUp[axialIndex][0], firstPairB = indexLookUp[axialIndex][1];
			glm::length_t secondPairA = indexLookUp[truncatedIndex][0], secondPairB = indexLookUp[truncatedIndex][1];


			obbProjections += this->halfs[firstPairA] * dotProducts[firstPairB][truncatedIndex] +
				this->halfs[firstPairB] * dotProducts[firstPairA][truncatedIndex];

			obbProjections += other.halfs[secondPairA] * dotProducts[axialIndex][secondPairB] +
				other.halfs[secondPairB] * dotProducts[axialIndex][secondPairA];
			obbProjections *= crossLengths[axialIndex][truncatedIndex];
			truncatedIndex += 2;
		}

		// In Case of Collision Whackiness break glass
		/*
		for (glm::length_t i = 0; i < 3; i++)
		{
			obbProjections += glm::abs(this->halfs[i] * glm::dot(glm::vec3(this->matrix[i]), axis));
			obbProjections += glm::abs(other.halfs[i] * glm::dot(glm::vec3(other.matrix[i]), axis));
		}

		if (!std::is_constant_evaluated())
		{
			if (glm::abs(obbProjections - testing) > EPSILON)
			{
				std::cout <<"Error in OBB Optimization: " <<  i << ":" << axialIndex << 
					":" << truncatedIndex << ":" << axis << ":" << obbProjections << "\t" << testing << std::endl;
			}
		}
		*/
		const float overlap = obbProjections - deltaProjection;
		// This axis is a separating one 
		if (deltaProjection > obbProjections)
		{
			return false;
		}
		// Find the minimum axis projection
		if (result.distance > overlap)
		{
			index = i;
			result.distance = overlap;
		}
	}
	// Annoying warning from implying index might not satisfy 0 <= index <= 15 - 1, when it can only be one of them
	// Result.normal is the direction this OBB needs to head to escape collision

#pragma warning( suppress : 28020 )
	result.normal = separatingAxes[index] * glm::sign(glm::dot(delta, separatingAxes[index]));
	result.point = this->Center() + result.distance * result.normal;
	return true;
}



// TODO: This isn't inline or constexpr i'm just too lazy to move it
// TODO: Allow response to be applied later, and in different amounts so you can have real physics tm(ie small object moves/rotates more than big one etc)
inline bool OrientedBoundingBox::OverlapAndSlide(const OrientedBoundingBox& other)
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

inline bool OrientedBoundingBox::Overlap(const Sphere& other) const
{
	Collision local;
	return this->Overlap(other, local);
}

inline float OrientedBoundingBox::ProjectionLength(const glm::vec3& vector) const
{
	float result = 0.f;
	result += glm::abs(glm::dot(glm::vec3(this->matrix[0]), vector)) * this->halfs[0];
	result += glm::abs(glm::dot(glm::vec3(this->matrix[1]), vector)) * this->halfs[1];
	result += glm::abs(glm::dot(glm::vec3(this->matrix[2]), vector)) * this->halfs[2];
	return result;
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
