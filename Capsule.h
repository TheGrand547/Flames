#pragma once
#ifndef CAPSULE_H
#define CAPSULE_H
#include <array>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "CollisionTypes.h"
#include "Lines.h"
#include "Sphere.h"

class Capsule
{
protected:
	LineSegment line;
	float radius;
public:
	Capsule() = default;
	constexpr Capsule(const Capsule& other) noexcept;
	constexpr Capsule(const Capsule&& other) noexcept;

	constexpr void SetCenter(const glm::vec3& center) noexcept;
	constexpr void Translate(const glm::vec3& delta) noexcept;
	constexpr void SetRadius(const float& value) noexcept;

	bool Intersect(const Capsule& other) const noexcept;
	/* 
	hit.depth will hold the distance that Other is inside of this Capsule
	hit.normal will hold the normal of the intersection(ie the axis to which moving along the reverse is not an intersection)
	hit.point will hold the point which is furthest into this capsule in Other
	*/
	bool Intersect(const Capsule& other, Collision& hit) const noexcept;

	bool Intersect(const Sphere& other) const noexcept;

	/*
	hit.depth will hold the distance that Other is inside of this Capsule
	hit.normal will hold the normal of the intersection(ie the axis to which moving along the reverse is not an intersection)
	hit.point will hold the point which is furthest into this capsule in Other
	*/
	bool Intersect(const Sphere& other, Collision& hit) const noexcept;

	inline constexpr float GetRadius() const noexcept;

	glm::vec3 ClosestPoint(const glm::vec3& other) const;

	void GenerateMesh(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18) const;

	static void GenerateMesh(Buffer<ArrayBuffer>& verts, Buffer<ElementArray>& indicies, float radius, float distance,
		const std::uint8_t latitudeSlices = 18, const std::uint8_t longitudeSlices = 18);
};

constexpr Capsule::Capsule(const Capsule& other) noexcept : line(other.line), radius(other.radius) {}

constexpr Capsule::Capsule(const Capsule&& other) noexcept : line(other.line), radius(other.radius) {}

constexpr void Capsule::SetCenter(const glm::vec3& center) noexcept
{
	const glm::vec3 direction = this->line.Direction() / 2.f;
	this->line.A = center + direction;
	this->line.B = center - direction;
}

constexpr void Capsule::Translate(const glm::vec3& delta) noexcept
{
	this->line.A += delta;
	this->line.B += delta;
}

constexpr void Capsule::SetRadius(const float& value) noexcept
{
	this->radius = value;
}

inline constexpr float Capsule::GetRadius() const noexcept
{
	return this->radius;
}

#endif // CAPSULE_H
