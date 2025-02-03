#pragma once
#ifndef BULLET_H
#define BULLET_H
#include "glmHelp.h"
#include "BasicPhysics.h"

struct Bullet
{
	static constexpr float Mass = 1.f;
	static constexpr float InvMass = 1.f / Mass;
	glm::vec3 position, velocity;

	void Update() noexcept;
};
#endif // BULLET_H