#pragma once
#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H
#include <array>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/glm.hpp>
#include <limits>
#include "AABB.h"

// TODO: EPSILON DUMB FUCK
#define EPSILON 0.000001f

class OrientedBoundingBox
{
private:
	// These are the basis vectors
	// TODO: maybe store only the euler orientation? trivial reconstruction of the basis
	glm::vec3 center;
	// TODO: Store these better, hack fraud
	std::array<std::pair<glm::vec3, float>, 3> axes;
public:
	constexpr OrientedBoundingBox();
	constexpr OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas = glm::vec3(1, 1, 1));
	constexpr OrientedBoundingBox(const OrientedBoundingBox& other) = default;
	constexpr OrientedBoundingBox(const AABB& other);
	~OrientedBoundingBox();

	inline constexpr void Rotate(const glm::vec3& euler);
	inline constexpr void Rotate(const glm::mat4& rotation);
	inline constexpr void Translate(const glm::vec3& distance);

	constexpr bool Intersect(glm::vec3 point, glm::vec3 dir, float& distance) const;
	constexpr bool Overlap(const OrientedBoundingBox& other) const;
	
	Model GetModel() const;
};

constexpr OrientedBoundingBox::OrientedBoundingBox() : center(0, 0, 0)
{

}

constexpr OrientedBoundingBox::OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas) : center(0, 0, 0)
{
	this->axes[0] = std::make_pair(glm::vec3(1, 0, 0), deltas.x);
	this->axes[1] = std::make_pair(glm::vec3(0, 1, 0), deltas.y);
	this->axes[2] = std::make_pair(glm::vec3(0, 0, 1), deltas.z);
	this->Rotate(glm::radians(euler));
}

constexpr OrientedBoundingBox::OrientedBoundingBox(const AABB& other)
{
	this->center = other.GetCenter();
	glm::vec3 temp = other.Deviation();

	this->axes[0] = std::make_pair(glm::vec3(1, 0, 0), temp.x);
	this->axes[1] = std::make_pair(glm::vec3(0, 1, 0), temp.y);
	this->axes[2] = std::make_pair(glm::vec3(0, 0, 1), temp.z);
}

inline constexpr void OrientedBoundingBox::Rotate(const glm::vec3& euler)
{
	this->Rotate(glm::eulerAngleXYZ(euler.x, euler.y, euler.z));
}

inline constexpr void OrientedBoundingBox::Rotate(const glm::mat4& rotation)
{
	for (auto& each : this->axes)
	{
		each.first = rotation * glm::vec4(each.first, 0);
	}
}

inline constexpr void OrientedBoundingBox::Translate(const glm::vec3& distance)
{
	this->center += distance;
}

// https://www.sciencedirect.com/topics/computer-science/oriented-bounding-box
constexpr bool OrientedBoundingBox::Intersect(glm::vec3 point, glm::vec3 dir, float& distance) const
{
	float nearDist = -std::numeric_limits<float>::infinity(), farDist = std::numeric_limits<float>::infinity();
	for (const auto& axis : this->axes)
	{
		float parallel = glm::dot(axis.first, this->center - point);
		if (glm::abs(glm::dot(dir, axis.first)) < EPSILON)
		{
			if (-parallel - axis.second > 0 || -parallel + axis.second > 0)
			{
				return false;
			}
		}

		float scaling = glm::dot(axis.first, dir);
		float param0 = (parallel + axis.second) / scaling;
		float param1 = (parallel - axis.second) / scaling;

		if (param0 > param1)
		{
			std::swap(param0, param1);
		}
		if (param0 > nearDist)
		{
			nearDist = param0;
		}
		if (param1 < farDist)
		{
			farDist = param1;
		}
		if (nearDist > farDist)
		{
			return false;
		}
		if (farDist < 0)
		{
			return false;
		}
	}
	if (nearDist > 0)
	{
		distance = nearDist;
	}
	else
	{
		distance = farDist;
	}
	return true;
}

// https://web.stanford.edu/class/cs273/refs/obb.pdf
constexpr bool OrientedBoundingBox::Overlap(const OrientedBoundingBox& other) const
{
	std::array<glm::vec3, 15> separatingAxes{};
	for (std::size_t i = 0; i < 3; i++)
	{
		separatingAxes[i * 5] = this->axes[i].first;
		separatingAxes[i * 5 + 1] = other.axes[i].first;
		for (std::size_t j = 0; j < 3; j++)
		{
			separatingAxes[i * 5 + 2 + j] = glm::cross(this->axes[i].first, other.axes[j].first);
		}
	}
	glm::vec3 delta = this->center - other.center;
	for (glm::vec3 axis : separatingAxes)
	{
		float left = glm::abs(glm::dot(axis, delta));
		float right = 0;
		for (std::size_t i = 0; i < 3; i++)
		{
			right += glm::abs(this->axes[i].second * glm::dot(this->axes[i].first, axis));
			right += glm::abs(other.axes[i].second * glm::dot(other.axes[i].first, axis));
		}
		// This axis is a separating one 
		if (left > right)
		{
			return false;
		}
	}
	return true;
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
