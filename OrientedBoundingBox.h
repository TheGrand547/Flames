#pragma once
#ifndef ORIENTED_BOUNDING_BOX_H
#define ORIENTED_BOUNDING_BOX_H
#include <array>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/glm.hpp>
#include <limits>
#include "AABB.h"
#include "Collidable.h"
#include "glmHelp.h"
#include "util.h"

class OrientedBoundingBox
{
private:
	// These are the basis vectors
	// TODO: maybe store only the euler orientation? trivial reconstruction of the basis
	glm::vec3 center;
	// TODO: Store these better, hack fraud
	std::array<std::pair<glm::vec3, float>, 3> axes;
	// The basis vectors can used to recreate the rotation matrix by simply transposing it
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
	inline constexpr glm::vec3 operator[](const std::size_t& t) const;
	inline constexpr glm::vec3 Center() const noexcept;


	inline constexpr void ReCenter(const glm::vec3& center) noexcept;
	inline constexpr void Reorient(const glm::vec3& euler);
	inline constexpr void Reorient(const glm::mat4& rotation);
	inline constexpr void Rotate(const glm::vec3& euler);
	inline constexpr void Rotate(const glm::mat4& rotation);
	inline constexpr void Scale(const glm::vec3& scale);
	inline constexpr void Translate(const glm::vec3& distance) noexcept;

	constexpr bool Intersect(const glm::vec3& origin, const glm::vec3& dir) const;
	// If no intersection is found, distance is undefined
	constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const;
	// If no intersection is found, result is undefined
	constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& result) const;
	// If no intersection is found, near and far hit are undefined
	constexpr bool Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& nearHit, Collision& farHit) const;
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
		deviation += glm::abs(axis.first) * axis.second;
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

inline constexpr glm::vec3 OrientedBoundingBox::operator[](const std::size_t& t) const
{
	assert(t < 3);
	return this->axes[t].first;
}

inline constexpr glm::vec3 OrientedBoundingBox::Center() const noexcept
{
	return this->center;
}

inline constexpr void OrientedBoundingBox::ReCenter(const glm::vec3& center) noexcept
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
		each.first = glm::normalize(rotation * glm::vec4(each.first, 0));
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

constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir) const
{
	float dist;
	return this->Intersect(point, dir, dist);
}

constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, float& distance) const
{
	Collision collision;
	bool value = this->Intersect(point, dir, collision);
	distance = collision.distance;
	return value;
}

constexpr bool OrientedBoundingBox::Intersect(const glm::vec3& point, const glm::vec3& dir, Collision& first) const
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
		if (param0 > nearHit.distance)
		{
			nearHit.distance = param0;
			nearHit.normal = axis.first;
		}
		if (param1 < farHit.distance)
		{
			farHit.distance = param1;
			farHit.normal = axis.first;
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
#include "util.h"
inline constexpr void OrientedBoundingBox::OverlapWithResponse(const OrientedBoundingBox& other, const glm::vec3 direction)
{
	std::array<glm::vec3, 15> separatingAxes{};
	for (std::size_t i = 0; i < 3; i++)
	{
		separatingAxes[i * 5] = this->axes[i].first;
		separatingAxes[i * 5 + 1] = other.axes[i].first;
		for (std::size_t j = 0; j < 3; j++)
		{
			separatingAxes[i * 5 + 2 + j] = glm::normalize(glm::cross(this->axes[i].first, other.axes[j].first));
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
		if (min > right - left)
		{
			index = i;
			min = right - left;
		}
	}
	//glm::vec3 normdir = glm::normalize(direction);
	glm::vec3 normdir = glm::normalize(-delta);
	std::cout << glm::dot(normdir, separatingAxes[index]) << ": " << min << ":" << separatingAxes[index] << ":" << normdir << std::endl;

	// min is the penetration depth? on axis separatingAxes[i]
	// dot(normdir, separatingAxes) > 0 they are aligned and must not be flipped or something


	//std::cout << "->" << separatingAxes[index] << ": " << min << ": " << index << std::endl;
	//std::cout << "<-" << direction << std::endl;
	//std::cout << "->" << min * separatingAxes[index] << std::endl;
	//this->center -= min * (normdir - SlideAlongPlane(separatingAxes[index], normdir)) * glm::length(direction); // what
	this->center += min * separatingAxes[index] * glm::sign(-glm::dot(normdir, separatingAxes[index]));
	//std::cout << SlideAlongPlane(separatingAxes[index], direction) << std::endl;

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
