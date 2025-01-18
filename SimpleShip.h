#pragma once
#ifndef SIMPLE_SHIP_H
#define SIMPLE_SHIP_H
#include "glmHelp.h"
#include "BasicPhysics.h"
#include "Transform.h"

class SimpleShip
{
protected:
	Transform location;
	BasicPhysics physics;

public:
	SimpleShip(const glm::vec3& position);
};

#endif // SIMPLE_SHIP_H
