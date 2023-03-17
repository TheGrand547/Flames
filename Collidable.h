#pragma once
#ifndef COLLIDABLE_H
#define COLLIDABLE_H
#include <glm/glm.hpp>
#include "AABB.h"

class Collidable
{
	virtual bool Collide() const = 0;
	virtual AABB GetBoundingBox() const = 0;
};

#endif // COLLIDABLE_H