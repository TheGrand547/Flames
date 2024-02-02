#pragma once
#ifndef AXIS_ALIGNED_BOUNDING_BOX_H
#define AXIS_ALIGNED_BOUNDING_BOX_H
#include <glm/glm.hpp>
#include <glm/ext/vector_common.hpp>
#include <vector>
#include "CollisionTypes.h"
#include "glmHelp.h"
#include "Model.h"
#include "util.h"

struct Sphere;

class AABB
{
//private:
//	const static std::array <glm::vec3, 3> Axes = { { glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1) } };
protected:
	glm::vec3 center, halfs;
public:
	constexpr AABB();
	constexpr AABB(const glm::vec3& dimension);
	constexpr AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound);
	constexpr AABB(const AABB& other);

	~AABB() = default;

	AABB& operator=(const AABB& other) noexcept;
	inline constexpr AABB operator+(const glm::vec3& point) const;
	inline constexpr AABB operator-(const glm::vec3& point) const;
	inline constexpr AABB& operator+=(const glm::vec3& point);
	inline constexpr AABB& operator-=(const glm::vec3& point);

	Model GetModel() const;
	
	inline constexpr float Volume() const;

	// Get the Center of the AABB
	inline constexpr glm::vec3 GetCenter() const;
	// Get the Half-Lengths of the AABB
	inline constexpr glm::vec3 Deviation() const;

	// Recenter the AABB
	inline constexpr void Center(const glm::vec3& point);
	inline constexpr void ScaleInPlace(const glm::vec3& factor);
	inline constexpr void ScaleInPlace(float x, float y = 1.0f, float z = 1.0f);
	// Move the AABB by factor in space
	inline constexpr void Translate(const glm::vec3& factor);

	inline constexpr bool PointInside(const glm::vec3& point) const;
	inline constexpr bool Overlap(const AABB& other) const;
	inline constexpr bool Contains(const AABB& other) const;

	// TODO: Other forms without both collisions y'know
	constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const;
	constexpr bool FastIntersect(const glm::vec3& point, const glm::vec3& dir) const;

	inline bool Overlap(const Sphere& other) const;
	bool Overlap(const Sphere& other, Collision& collision) const;

	static constexpr AABB MakeAABB(const glm::vec3& left, const glm::vec3& right);
	static constexpr AABB MakeAABB(const std::vector<glm::vec3>& points);
	template<IsVec3... Args> inline static constexpr AABB MakeAABB(const Args... args);

	constexpr bool operator==(const AABB& other) const
	{
		return this->center == other.center && this->halfs == other.halfs;
	}
	constexpr bool operator!=(const AABB& other) const
	{
		return !(*this == other);
	}
};

constexpr AABB::AABB() : center(0), halfs(1)
{

}

constexpr AABB::AABB(const glm::vec3& dimension) : center(0), halfs(glm::abs(dimension) / 2.f)
{

}

constexpr AABB::AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : center((negativeBound + positiveBound) / 2.f), 
																						halfs(glm::abs(negativeBound - positiveBound) / 2.f)
{

}

constexpr AABB::AABB(const AABB& other) : center(other.center), halfs(other.halfs)
{

}

inline constexpr float AABB::Volume() const
{
	return this->halfs.x * this->halfs.y * this->halfs.z * 8.f;
}

inline constexpr glm::vec3 AABB::GetCenter() const
{
	return this->center;
}

inline constexpr glm::vec3 AABB::Deviation() const
{
	return this->halfs;
}

inline constexpr AABB AABB::operator+(const glm::vec3& point) const
{
	return AABB(this->center + point, this->halfs);
}

inline constexpr AABB AABB::operator-(const glm::vec3& point) const
{
	return AABB(this->center - point, this->halfs);
}

inline constexpr AABB& AABB::operator+=(const glm::vec3& point) 
{
	this->center += point;
	return *this;
}

inline constexpr AABB& AABB::operator-=(const glm::vec3& point)
{
	this->center -= point;
	return *this;
}

inline constexpr void AABB::Center(const glm::vec3& point)
{
	this->center = point;
}

inline constexpr void AABB::ScaleInPlace(const glm::vec3& scale)
{
	this->halfs *= glm::abs(scale);
}

inline constexpr void AABB::ScaleInPlace(float x, float y, float z)
{
	this->ScaleInPlace(glm::vec3(x, y, z));
}

inline constexpr void AABB::Translate(const glm::vec3& point)
{
	this->center += point;
}


inline constexpr bool AABB::PointInside(const glm::vec3& point) const
{
	glm::vec3 negativeBound = this->center - this->halfs, positiveBound = this->center + this->halfs;
	bool xInside = negativeBound.x <= point.x && point.x <= positiveBound.x;
	bool yInside = negativeBound.y <= point.y && point.y <= positiveBound.y;
	bool zInside = negativeBound.z <= point.z && point.z <= positiveBound.z;
	//glm::all(glm::lessThanEqual(negativeBound, point)) && glm::all(glm::lessThanEqual(point, positiveBound))
	return xInside && yInside && zInside;
}

inline constexpr bool AABB::Overlap(const AABB& other) const
{
	glm::vec3 negativeBound = this->center - this->halfs, positiveBound = this->center + this->halfs;
	glm::vec3 negativeBoundOther = other.center - other.halfs, positiveBoundOther = other.center + other.halfs;

	bool xInside = negativeBoundOther.x <= positiveBound.x && positiveBoundOther.x >= negativeBound.x;
	bool yInside = negativeBoundOther.y <= positiveBound.y && positiveBoundOther.y >= negativeBound.y;
	bool zInside = negativeBoundOther.z <= positiveBound.z && positiveBoundOther.z >= negativeBound.z;
	return xInside && yInside && zInside;
}

inline constexpr bool AABB::Contains(const AABB& other) const
{
	glm::vec3 negativeBound = this->center - this->halfs, positiveBound = this->center + this->halfs;
	glm::vec3 negativeBoundOther = other.center - other.halfs, positiveBoundOther = other.center + other.halfs;

	bool xInside = negativeBound.x <= negativeBoundOther.x && positiveBound.x >= positiveBoundOther.x;
	bool yInside = negativeBound.y <= negativeBoundOther.y && positiveBound.y >= positiveBoundOther.y;
	bool zInside = negativeBound.z <= negativeBoundOther.z && positiveBound.z >= positiveBoundOther.z;
	return xInside && yInside && zInside;
}

inline bool AABB::Overlap(const Sphere& other) const
{
	Collision collide{};
	return this->Overlap(other, collide);
}

// Modified version of the OBB code to be in theory "better", ie faster
constexpr bool AABB::Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const
{
	nearHit.Clear();
	farHit.Clear();
	nearHit.distance = -std::numeric_limits<float>::infinity();
	farHit.distance = std::numeric_limits<float>::infinity();

	glm::vec3 direction = this->center - point;

	for (auto i = 0; i < 3; i++)
	{
		float scale = this->halfs[i];
		float parallel = direction[i];
		if (glm::abs(dir[i]) < EPSILON)
		{
			//if (-parallel - scale > 0 || -parallel + scale > 0)
			if (abs(parallel) > scale)
			{
				return false;
			}
		}

		float scaling = dir[i];
		float param0 = (parallel + scale) / scaling;
		float param1 = (parallel - scale) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearHit.distance)
		{
			nearHit.distance = param0;
			nearHit.normal = glm::vec3(i == 0, i == 1, i == 2) * glm::sign(-parallel);
		}
		if (param1 < farHit.distance)
		{
			farHit.distance = param1;
			farHit.normal = glm::vec3(i == 0, i == 1, i == 2) * glm::sign(-parallel);
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

inline constexpr bool AABB::FastIntersect(const glm::vec3& point, const glm::vec3& dir) const
{
	glm::vec3 delta = this->center - point;
	float nearHit = -std::numeric_limits<float>::infinity(), farHit = std::numeric_limits<float>::infinity();

	for (auto i = 0; i < 3; i++)
	{
		float scale = this->halfs[i];
		float parallel = delta[i];
		if (glm::abs(dir[i]) < EPSILON)
		{
			//if (-parallel - scale > 0 || -parallel + scale > 0)
			if (abs(parallel) > scale)
			{
				return false;
			}
		}

		float scaling = dir[i];
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

constexpr AABB AABB::MakeAABB(const glm::vec3& left, const glm::vec3& right)
{
	return AABB(glm::fmin(left, right), glm::fmax(left, right));
}

constexpr AABB AABB::MakeAABB(const std::vector<glm::vec3>& points)
{
	glm::vec3 min(INFINITY, INFINITY, INFINITY), max(-INFINITY, -INFINITY, -INFINITY);
	for (int i = 0; i < points.size(); i++)
	{
		min = glm::fmin(min, points[i]);
		max = glm::fmax(max, points[i]);
	}
	return AABB(min, max);
}

template<IsVec3 ...Args>
inline constexpr AABB AABB::MakeAABB(const Args ...args)
{
	std::vector<glm::vec3> points = { args... };
	return AABB::MakeAABB(points);
}

typedef AABB AxisAlignedBox;

#endif // AXIS_ALIGNED_BOUNDING_BOX_H

