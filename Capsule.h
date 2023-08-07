#pragma once
#ifndef CAPSULE_H
#define CAPSULE_H
#include <array>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "Lines.h"
#include "Collidable.h"

class Capsule
{
protected:
	LineSegment line;
	float radius;
public:


	bool Intersect(const Capsule& other) const noexcept;
	bool Intersect(const Capsule& other, Collision& hit) const noexcept;
};

#endif // CAPSULE_H
