#pragma once
#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H
#include <array>
#include <glm/gtc/epsilon.hpp>
#include <glm/glm.hpp>
#include "AABB.h"

// TODO: EPSILON DUMB FUCK
#define EPSILON 0.000001

class OrientedBoundingBox
{
private:
	// These are the basis vectors
	glm::vec3 center; // Maybe center?
	std::array<std::pair<glm::vec3, float>, 3> axes;
public:
	constexpr OrientedBoundingBox();
	~OrientedBoundingBox();

	inline constexpr void Translate(const glm::vec3& distance);

	constexpr bool intersect(glm::vec3 point, glm::vec3 dir, float& distance) const;
	constexpr bool overlap(const OrientedBoundingBox& other) const;
};

inline constexpr void OrientedBoundingBox::Translate(const glm::vec3& distance)
{
	this->center += distance;
}

// https://www.sciencedirect.com/topics/computer-science/oriented-bounding-box
constexpr bool OrientedBoundingBox::intersect(glm::vec3 point, glm::vec3 dir, float& distance) const
{
	float near = -INFINITY, far = INFINITY;

	for (const auto& axis : this->axes)
	{
		float r = glm::dot(axis.first, this->center - point);
		if (glm::abs(glm::dot(dir, axis.first)) < EPSILON)
		{
			if (-r - axis.second > 0 || -r + axis.second > 0)
			{
				return false;
			}
		}

		float s = glm::dot(axis.first, dir);
		float t0 = (r + axis.second) / s;
		float t1 = (r - axis.second) / s;

		if (t0 > t1)
		{
			std::swap(t0, t1);
		}
		if (t0 > near)
		{
			near = t0;
		}
		if (t1 < far)
		{
			far = t1;
		}
		if (near > far)
		{
			return false;
		}
		if (far < 0)
		{
			return false;
		}
	}
	if (near > 0)
	{
		distance = near;
	}
	else
	{
		distance = far;
	}
	return true;
}

// https://web.stanford.edu/class/cs273/refs/obb.pdf
constexpr bool OrientedBoundingBox::overlap(const OrientedBoundingBox& other) const
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
#endif // ORIENTED_BOUNDING_BOX_H
