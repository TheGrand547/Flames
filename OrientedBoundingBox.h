#pragma once
#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H
#include <array>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/glm.hpp>
#include <limits>
#include "AABB.h"
#include "glmHelp.h"

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
	constexpr OrientedBoundingBox(const glm::vec3& euler = glm::vec3(0, 0, 0), const glm::vec3& deltas = glm::vec3(1, 1, 1));
	constexpr OrientedBoundingBox(const OrientedBoundingBox& other) = default;
	constexpr OrientedBoundingBox(const Model& model);
	constexpr OrientedBoundingBox(const AABB& other);
	~OrientedBoundingBox() = default;

	OrientedBoundingBox& operator=(const OrientedBoundingBox& other) = default;

	inline constexpr AABB GetAABB() const noexcept;

	inline constexpr glm::vec3 Forward() const noexcept;
	inline constexpr glm::vec3 Up() const noexcept;
	inline constexpr glm::vec3 Cross() const noexcept;

	inline constexpr void Center(const glm::vec3& center) noexcept;
	inline constexpr void Reorient(const glm::vec3& euler);
	inline constexpr void Reorient(const glm::mat4& rotation);
	inline constexpr void Rotate(const glm::vec3& euler);
	inline constexpr void Rotate(const glm::mat4& rotation);
	inline constexpr void Scale(const glm::vec3& scale);
	inline constexpr void Translate(const glm::vec3& distance) noexcept;

	constexpr bool Intersect(glm::vec3 point, glm::vec3 dir, float& distance) const;
	constexpr bool Overlap(const OrientedBoundingBox& other) const;
	
	constexpr void OverlapWithResponse(const OrientedBoundingBox& other, const glm::vec3 dir = glm::vec3(0));

	inline Model GetModel() const;
};

constexpr OrientedBoundingBox::OrientedBoundingBox(const glm::vec3& euler, const glm::vec3& deltas) : center(0, 0, 0)
{
	this->axes[0] = std::make_pair(glm::vec3(1, 0, 0), deltas.x);
	this->axes[1] = std::make_pair(glm::vec3(0, 1, 0), deltas.y);
	this->axes[2] = std::make_pair(glm::vec3(0, 0, 1), deltas.z);
	this->Rotate(euler);
}

constexpr OrientedBoundingBox::OrientedBoundingBox(const Model& model) : OrientedBoundingBox(model.rotation, model.scale)
{
	this->center = model.translation;
}

constexpr OrientedBoundingBox::OrientedBoundingBox(const AABB& other)
{
	this->center = other.GetCenter();
	glm::vec3 temp = other.Deviation();

	this->axes[0] = std::make_pair(glm::vec3(1, 0, 0), temp.x);
	this->axes[1] = std::make_pair(glm::vec3(0, 1, 0), temp.y);
	this->axes[2] = std::make_pair(glm::vec3(0, 0, 1), temp.z);
}

inline Model OrientedBoundingBox::GetModel() const
{
	glm::mat4 mat(glm::vec4(this->axes[0].first, 0), glm::vec4(this->axes[1].first, 0),
		glm::vec4(this->axes[2].first, 0), glm::vec4(0, 0, 0, 1));
	glm::vec3 angles{ 0.f, 0.f, 0.f };
	glm::extractEulerAngleXYZ(mat, angles.x, angles.y, angles.z);
	return Model(this->center, glm::degrees(angles), glm::vec3(this->axes[0].second, this->axes[1].second, this->axes[2].second));
}

inline constexpr AABB OrientedBoundingBox::GetAABB() const noexcept
{
	glm::vec3 deviation(0.f);
	for (const auto& axis : this->axes)
	{
		deviation += axis.first * axis.second;
	}
	return AABB(this->center - deviation, this->center + deviation);
}

inline constexpr glm::vec3 OrientedBoundingBox::Forward() const noexcept
{
	return this->axes[0].first;
}

inline constexpr glm::vec3 OrientedBoundingBox::Up() const noexcept
{
	return this->axes[1].first;
}

inline constexpr glm::vec3 OrientedBoundingBox::Cross() const noexcept
{
	return this->axes[2].first;
}

inline constexpr void OrientedBoundingBox::Center(const glm::vec3& center) noexcept
{
	this->center = center;
}

inline constexpr void OrientedBoundingBox::Reorient(const glm::vec3& euler)
{
	this->Reorient(glm::eulerAngleXYZ(euler.x, euler.y, euler.z));
}

inline constexpr void OrientedBoundingBox::Reorient(const glm::mat4& rotation)
{
	this->axes[0].first = glm::vec3(1, 0, 0);
	this->axes[1].first = glm::vec3(0, 1, 0);
	this->axes[2].first = glm::vec3(0, 0, 1);
	this->Rotate(rotation);
}

inline constexpr void OrientedBoundingBox::Rotate(const glm::vec3& euler)
{
	this->Rotate(glm::eulerAngleXYZ(glm::radians(euler.x), glm::radians(euler.y), glm::radians(euler.z)));
}

inline constexpr void OrientedBoundingBox::Rotate(const glm::mat4& rotation)
{
	for (auto& each : this->axes)
	{
		each.first = rotation * glm::vec4(each.first, 0);
	}
}

inline constexpr void OrientedBoundingBox::Scale(const glm::vec3& scale)
{
	for (int i = 0; i < 3; i++)
	{
		this->axes[i].second *= scale[i];
	}
}

inline constexpr void OrientedBoundingBox::Translate(const glm::vec3& distance) noexcept
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
#include <iostream>
inline constexpr void OrientedBoundingBox::OverlapWithResponse(const OrientedBoundingBox& other, const glm::vec3 direction)
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
	float min = INFINITY;
	std::size_t index = 0;
	for (std::size_t i = 0; i < separatingAxes.size(); i++)
	{
		glm::vec3 axis = separatingAxes[i];
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
			return;
		}
		if (min > left - right)
		{
			index = i;
			min = left - right;
		}
	}
	// Minimum separating axis is separatingAxes[i]
	// fumo
	//std::array<glm::vec3, 3> basis;
	//std::cout << min << " : " << separatingAxes[index] << std::endl;
	//this->center += separatingAxes[index] * min;
	/*
	// One of the cross product ones
	if (index % 5 > 1)
	{
		int intern = index % 5;
		basis[0] = this->axes[index / 5].first;
		basis[1] = other.axes[(index % 5) - 2].first;
		basis[2] = separatingAxes[index];
	}
	std::cout << this->center << std::endl;
	*/
	float dot = 0;
	std::size_t dotIndex = 0;

	glm::vec3 dir;
	float len = 1;
	if (glm::length(direction) > 0.0001)
		len = glm::length(direction);
	else
		return;
	dir = glm::normalize((direction.x == direction.y && direction.y == 0 && direction.z == direction.y) ?  other.center - this->center : direction);
	std::cout << "DIR: " << dir << ": " << direction.length() << std::endl;
	// Find most aligned vector
	for (std::size_t i = 0; i < 3; i++)
	{
		std::cout << "DOT " << other.axes[i].first << ": " << glm::dot(dir, glm::normalize(other.axes[i].first)) << std::endl;
		if (glm::abs(glm::dot(dir, glm::normalize(other.axes[i].first)) > glm::abs(dot)))
		{
			dotIndex = i;
			dot = glm::dot(dir, glm::normalize(other.axes[i].first));
		}
	}
	std::cout << "PROJ: " << other.axes[dotIndex].first << ":" << dot << std::endl;
	dot = -glm::sign(dot);
	//this->center += len * (glm::normalize(glm::reflect(dir, dot * other.axes[dotIndex].first)) - dir) / 20.f ;
	this->center -= len * dir;
	/*
	if (!this->Overlap(other))
		return;
	glm::vec3 normal = glm::normalize(this->center - other.center);
	int mostAligned = 0, theirAligned = 0;
	float leftRadius = -INFINITY, rightRadius = -INFINITY;
	for (int i = 0; i < 3; i++)
	{
		leftRadius = glm::max(leftRadius, glm::abs(this->axes[i].second));
		rightRadius = glm::max(rightRadius, glm::abs(other.axes[i].second));
		// I think this was an idea but I'm not sure of what quality, need to test it
		if (glm::abs(glm::dot(this->axes[i].first, normal)) > glm::abs(glm::dot(this->axes[mostAligned].first, normal)))
			mostAligned = i;
		if (glm::abs(glm::dot(other.axes[i].first, normal)) > glm::abs(glm::dot(other.axes[theirAligned].first, normal)))
			theirAligned = i;
		leftRadius = this->axes[mostAligned].second, rightRadius = other.axes[theirAligned].second;
	}
	float sum = leftRadius + rightRadius;
	this->center = other.center + sum * normal;
	*/
	// Project them to AABB in one of their coordinate systems via change of basis
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
