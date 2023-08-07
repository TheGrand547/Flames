#pragma once
#ifndef AXIS_ALIGNED_BOUNDING_BOX_H
#define AXIS_ALIGNED_BOUNDING_BOX_H
#include <glm/glm.hpp>
#include <vector>
#include "Model.h"

class AABB
{
protected:
	glm::vec3 negativeBound, positiveBound;
public:
	constexpr AABB();
	constexpr AABB(const glm::vec3& dimension);
	constexpr AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound);
	constexpr AABB(const AABB& other);

	~AABB() = default;

	AABB& operator=(const AABB& other) noexcept;
	inline constexpr AABB operator+(const glm::vec3& point) const;

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

	static constexpr AABB MakeAABB(const glm::vec3& left, const glm::vec3& right);
	static constexpr AABB MakeAABB(const std::vector<glm::vec3>& points);

	bool operator==(const AABB& other) const
	{
		return negativeBound == other.negativeBound && positiveBound == other.positiveBound;
	}
};

constexpr AABB::AABB() : negativeBound(-1), positiveBound(1)
{

}

constexpr AABB::AABB(const glm::vec3& dimension) : negativeBound(-glm::abs(dimension) / 2.f), positiveBound(glm::abs(dimension) / 2.f)
{

}

constexpr AABB::AABB(const glm::vec3& negativeBound, const glm::vec3& positiveBound) : negativeBound(negativeBound), positiveBound(positiveBound)
{

}

constexpr AABB::AABB(const AABB& other)
{
	this->negativeBound = other.negativeBound;
	this->positiveBound = other.positiveBound;
}

inline constexpr float AABB::Volume() const
{
	glm::vec3 deviation(this->Deviation());
	return deviation.x * deviation.y * deviation.z * 8.f;
}

inline constexpr glm::vec3 AABB::GetCenter() const
{
	return (this->negativeBound + this->positiveBound) / 2.f;
}

inline constexpr glm::vec3 AABB::Deviation() const
{
	return glm::abs((this->positiveBound - this->negativeBound) / 2.f);
}

inline constexpr AABB AABB::operator+(const glm::vec3& point) const
{
	return AABB(this->negativeBound + point, this->positiveBound + point);
}

inline constexpr void AABB::Center(const glm::vec3& point)
{
	glm::vec3 delta = (this->positiveBound - this->negativeBound) / 2.0f;
	this->negativeBound = point - delta;
	this->positiveBound = point + delta;
}

inline constexpr void AABB::ScaleInPlace(const glm::vec3& scale)
{
	glm::vec3 center = (this->positiveBound + this->negativeBound) / 2.0f;
	this->positiveBound = (this->positiveBound - this->negativeBound) * scale;
	this->Center(center);
}

inline constexpr void AABB::ScaleInPlace(float x, float y, float z)
{
	this->ScaleInPlace(glm::vec3(x, y, z));
}

inline constexpr void AABB::Translate(const glm::vec3& point)
{
	this->positiveBound += point;
	this->negativeBound += point;
}


inline constexpr bool AABB::PointInside(const glm::vec3& point) const
{
	bool xInside = this->negativeBound.x < point.x && point.x < this->positiveBound.x;
	bool yInside = this->negativeBound.y < point.y && point.y < this->positiveBound.y;
	bool zInside = this->negativeBound.z < point.z && point.z < this->positiveBound.z;
	return xInside && yInside && zInside;
}

inline constexpr bool AABB::Overlap(const AABB& other) const
{
	bool xInside = this->negativeBound.x < other.positiveBound.x && this->positiveBound.x > other.negativeBound.x;
	bool yInside = this->negativeBound.y < other.positiveBound.y && this->positiveBound.y > other.negativeBound.y;
	bool zInside = this->negativeBound.z < other.positiveBound.z && this->positiveBound.z > other.negativeBound.z;
	return xInside && yInside && zInside;
}

inline constexpr bool AABB::Contains(const AABB& other) const
{
	// TODO: This is bad and slow
	return this->PointInside(other.GetCenter() + other.Deviation()) && this->PointInside(other.GetCenter() - other.Deviation());
}

constexpr AABB AABB::MakeAABB(const glm::vec3& left, const glm::vec3& right)
{
	glm::vec3 min{}, max{};

	min.x = glm::min(left.x, right.x);
	min.y = glm::min(left.y, right.y);
	min.z = glm::min(left.z, right.z);

	max.x = glm::max(left.x, right.x);
	max.y = glm::max(left.y, right.y);
	max.z = glm::max(left.z, right.z);

	return AABB(min, max);
}

constexpr AABB AABB::MakeAABB(const std::vector<glm::vec3>& points)
{
	glm::vec3 min(INFINITY, INFINITY, INFINITY), max(-INFINITY, -INFINITY, -INFINITY);
	for (int i = 0; i < points.size(); i++)
	{
		glm::vec3 point = points[i];
		min.x = glm::min(min.x, point.x);
		min.y = glm::min(min.y, point.y);
		min.z = glm::min(min.z, point.z);

		max.x = glm::max(max.x, point.x);
		max.y = glm::max(max.y, point.y);
		max.z = glm::max(max.z, point.z);
	}
	return AABB(min, max);
}

typedef AABB AxisAlignedBox;

#endif // AXIS_ALIGNED_BOUNDING_BOX_H