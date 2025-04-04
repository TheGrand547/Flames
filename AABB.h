#pragma once
#ifndef AXIS_ALIGNED_BOUNDING_BOX_H
#define AXIS_ALIGNED_BOUNDING_BOX_H
#include <glm/glm.hpp>
#include <glm/ext/vector_common.hpp>
#include <glm/gtx/component_wise.hpp>
#include <vector>
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Model.h"
#include "util.h"
#include "log.h"
#include "Lines.h"

struct Sphere;

class AABB
{
protected:
	glm::vec3 center, halfs;
public:
	inline AABB() noexcept;
	inline AABB(const glm::vec3& sideLengths) noexcept;
	inline AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound) noexcept;
	inline AABB(const AABB& other) noexcept;

	~AABB() noexcept = default;

	AABB& operator=(const AABB& other) noexcept;
	inline AABB operator+(const glm::vec3& point) const noexcept;
	inline AABB operator-(const glm::vec3& point) const noexcept;
	inline AABB& operator+=(const glm::vec3& point) noexcept;
	inline AABB& operator-=(const glm::vec3& point) noexcept;

	Model GetModel() const noexcept;
	glm::mat4 GetModelMatrix() const noexcept;
	glm::mat4 GetNormalMatrix() const noexcept;
	
	inline float Volume() const noexcept;

	inline float SignedDistance(const glm::vec3& point) const noexcept;

	// Get the Center of the AABB
	inline glm::vec3 GetCenter() const noexcept;

	// Get the Half-Lengths of the AABB
	inline glm::vec3 Deviation() const noexcept;

	// Recenter the AABB
	inline void SetCenter(const glm::vec3& point) noexcept;
	inline void SetScale(const glm::vec3& factor) noexcept;
	inline void SetScale(const float& value) noexcept;
	inline void ScaleInPlace(const glm::vec3& factor) noexcept;
	inline void ScaleInPlace(float x, float y = 1.0f, float z = 1.0f) noexcept;
	// Move the AABB by factor in space
	inline void Translate(const glm::vec3& factor) noexcept;

	inline bool PointInside(const glm::vec3& point) const noexcept;
	inline bool Overlap(const AABB& other) const noexcept;
	inline bool Contains(const AABB& other) const noexcept;

	// Don't use these unless you *really* need the collision normals
	inline bool Intersect(const glm::vec3& point, const glm::vec3& dir) const noexcept;
	inline bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit) const noexcept;
	bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const noexcept;

	inline bool FastIntersect(const glm::vec3& point, const glm::vec3& dir) const noexcept;
	bool FastIntersect(const glm::vec3& point, const glm::vec3& dir, float& near, float& far) const noexcept;

	bool FastIntersect(Ray ray) const noexcept;
	inline bool FastIntersect(LineSegment line) const noexcept;
	inline bool FastIntersect(Line line) const noexcept;

	inline bool Overlap(const Sphere& other) const noexcept;
	bool Overlap(const Sphere& other, Collision& collision) const noexcept;

	inline static AABB MakeAABB(const glm::vec3& left, const glm::vec3& right) noexcept;
	inline static AABB MakeAABB(const std::vector<glm::vec3>& points) noexcept;
	inline static AABB CombineAABB(const AABB& A, const AABB& B) noexcept;
	template<IsVec3... Args> inline static AABB MakeAABB(const Args... args) noexcept;


	inline bool operator==(const AABB& other) const noexcept
	{
		return this->center == other.center && this->halfs == other.halfs;
	}

	inline bool operator!=(const AABB& other) const noexcept
	{
		return !(*this == other);
	}
};

inline AABB::AABB() noexcept : center(0), halfs(1)
{

}

inline AABB::AABB(const glm::vec3& sideLengths) noexcept : center(0), halfs(glm::abs(sideLengths) / 2.f)
{

}

inline  AABB::AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound) noexcept : center((negativeBound + positiveBound) / 2.f),
																						halfs(glm::abs(negativeBound - positiveBound) / 2.f)
{

}

inline AABB::AABB(const AABB& other) noexcept : center(other.center), halfs(other.halfs)
{

}

inline float AABB::Volume() const noexcept
{
	return glm::compMul(this->halfs) * 8.f;
}

// From https://iquilezles.org/articles/distfunctions/
inline float AABB::SignedDistance(const glm::vec3& point) const noexcept
{
	glm::vec3 transformed = glm::abs(point - this->center) - this->halfs;
	return glm::length(glm::max(transformed, glm::vec3(0.f))) + glm::min(glm::compMax(transformed), 0.f);
}

inline glm::vec3 AABB::GetCenter() const noexcept
{
	return this->center;
}

inline glm::vec3 AABB::Deviation() const noexcept
{
	return this->halfs;
}

inline AABB AABB::operator+(const glm::vec3& point) const noexcept
{
	return AABB(this->center + point, this->halfs);
}

inline AABB AABB::operator-(const glm::vec3& point) const noexcept
{
	return AABB(this->center - point, this->halfs);
}

inline AABB& AABB::operator+=(const glm::vec3& point) noexcept
{
	this->center += point;
	return *this;
}

inline AABB& AABB::operator-=(const glm::vec3& point) noexcept
{
	this->center -= point;
	return *this;
}

inline void AABB::SetCenter(const glm::vec3& point) noexcept
{
	this->center = point;
}

inline void AABB::SetScale(const glm::vec3& factor) noexcept
{
	this->halfs = factor;
}

inline void AABB::SetScale(const float& scale) noexcept
{
	this->halfs = glm::vec3(scale);
}

inline void AABB::ScaleInPlace(const glm::vec3& scale) noexcept
{
	this->halfs *= glm::abs(scale);
}

inline void AABB::ScaleInPlace(float x, float y, float z) noexcept
{
	this->ScaleInPlace(glm::vec3(x, y, z));
}

inline void AABB::Translate(const glm::vec3& point) noexcept
{
	this->center += point;
}

inline bool AABB::PointInside(const glm::vec3& point) const noexcept
{
	glm::vec3 negativeBound = this->center - this->halfs, positiveBound = this->center + this->halfs;
	bool result = glm::all(glm::lessThanEqual(negativeBound, point)) && glm::all(glm::lessThanEqual(point, positiveBound));
#ifdef _DEBUG
	bool xInside = negativeBound.x <= point.x && point.x <= positiveBound.x;
	bool yInside = negativeBound.y <= point.y && point.y <= positiveBound.y;
	bool zInside = negativeBound.z <= point.z && point.z <= positiveBound.z;
	if (result != (xInside && yInside && zInside))
	{
		Log("Shortcut for Point in AABB failed");
		result = xInside && yInside && zInside;
	}
#endif // !_DEBUG
	return result;
}

inline bool AABB::Overlap(const AABB& other) const noexcept
{
	glm::vec3 negativeBound = this->center - this->halfs, positiveBound = this->center + this->halfs;
	glm::vec3 negativeBoundOther = other.center - other.halfs, positiveBoundOther = other.center + other.halfs;
	bool result = glm::all(glm::lessThanEqual(negativeBoundOther, positiveBound)) && glm::all(glm::lessThanEqual(negativeBound, positiveBoundOther));
#ifdef _DEBUG
	bool xInside = negativeBoundOther.x <= positiveBound.x && positiveBoundOther.x >= negativeBound.x;
	bool yInside = negativeBoundOther.y <= positiveBound.y && positiveBoundOther.y >= negativeBound.y;
	bool zInside = negativeBoundOther.z <= positiveBound.z && positiveBoundOther.z >= negativeBound.z;
	if (result != (xInside && yInside && zInside))
	{
		Log("Shortcut for AABB overlap AABB failed");
		result = xInside && yInside && zInside;
	}
#endif // !_DEBUG
	return result;
}

inline bool AABB::Contains(const AABB& other) const noexcept
{
	glm::vec3 negativeBound = this->center - this->halfs, positiveBound = this->center + this->halfs;
	glm::vec3 negativeBoundOther = other.center - other.halfs, positiveBoundOther = other.center + other.halfs;
	bool result = glm::all(glm::lessThanEqual(negativeBound, negativeBoundOther)) && glm::all(glm::lessThanEqual(positiveBoundOther, positiveBound));
#ifdef _DEBUG
	bool xInside = negativeBound.x <= negativeBoundOther.x && positiveBound.x >= positiveBoundOther.x;
	bool yInside = negativeBound.y <= negativeBoundOther.y && positiveBound.y >= positiveBoundOther.y;
	bool zInside = negativeBound.z <= negativeBoundOther.z && positiveBound.z >= positiveBoundOther.z;
	if (result != (xInside && yInside && zInside))
	{
		Log("Shortcut for AABB inside AABB failed");
	}
	result = xInside && yInside && zInside;
#endif // !_DEBUG
	return result;
}

inline bool AABB::Overlap(const Sphere& other) const noexcept
{
	Collision collide{};
	return this->Overlap(other, collide);
}

inline bool AABB::Intersect(const glm::vec3& point, const glm::vec3& dir) const noexcept
{
	Collision near{}, far{};

	return this->Intersect(point, dir, near, far);
}

inline bool AABB::Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& near) const noexcept
{
	Collision far{};
	return this->Intersect(point, dir, near, far);
}

inline bool AABB::FastIntersect(Line line) const noexcept
{
	return this->FastIntersect(line.PointA(), line.delta);
}

inline bool AABB::FastIntersect(LineSegment segment) const noexcept
{
	float near = -INFINITY, far = INFINITY;
	bool result = this->FastIntersect(segment.A, segment.Direction(), near, far);
	return result && ((far >= 0.f && far <= 1.f) || (near >= 0.f && near <= 1.f));
}

inline bool AABB::FastIntersect(const glm::vec3& point, const glm::vec3& dir) const noexcept
{
	float dummyA = 0.f, dummyB = 0.f;
	return this->FastIntersect(point, dir, dummyA, dummyB);
}

inline AABB AABB::MakeAABB(const glm::vec3& left, const glm::vec3& right) noexcept
{
	return AABB(glm::fmin(left, right), glm::fmax(left, right));
}

inline AABB AABB::MakeAABB(const std::vector<glm::vec3>& points) noexcept
{
	if (points.size() > 0)
	{
		glm::vec3 min(points[0]), max(points[0]);
		for (std::size_t i = 1; i < points.size(); i++)
		{
			min = glm::fmin(min, points[i]);
			max = glm::fmax(max, points[i]);
		}
		return AABB(min, max);
	}
	return AABB(glm::vec3(-INFINITY), glm::vec3(INFINITY));
}

inline AABB AABB::CombineAABB(const AABB& A, const AABB& B) noexcept
{
	glm::vec3 lowA = A.center - A.halfs, highA = A.center + A.halfs;
	glm::vec3 lowB = B.center - B.halfs, highB = B.center + B.halfs;
	return AABB(glm::fmin(lowA, lowB), glm::fmax(highA, highB));
}

template<IsVec3 ...Args>
inline AABB AABB::MakeAABB(const Args ...args) noexcept
{
	std::vector<glm::vec3> points = { args... };
	return AABB::MakeAABB(points);
}

template<typename T>
	requires requires(const T& element)
{
	{ element.GetAABB() } -> std::convertible_to<AABB>;
}
inline AABB GetAABB(const T& type)
{
	return type.GetAABB();
}

template<typename T>
	requires requires(const T*& element)
{
	{ element->GetAABB() } -> std::convertible_to<AABB>;
}
inline AABB GetAABB(const T*& type)
{
	return type->GetAABB();
}


typedef AABB AxisAlignedBox;

#endif // AXIS_ALIGNED_BOUNDING_BOX_H

