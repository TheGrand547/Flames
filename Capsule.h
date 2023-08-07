#pragma once
#ifndef CAPSULE_H
#define CAPSULE_H
#include <array>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "Collidable.h"
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
};

constexpr Capsule::Capsule(const Capsule& other) noexcept : line(other.line), radius(radius) {}

constexpr Capsule::Capsule(const Capsule&& other) noexcept : line(other.line), radius(radius) {}

#endif // CAPSULE_H
