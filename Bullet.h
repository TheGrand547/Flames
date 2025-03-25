#pragma once
#ifndef BULLET_H
#define BULLET_H
#include "glmHelp.h"
#include "BasicPhysics.h"
#include "AABB.h"

struct Bullet
{
	static constexpr float Mass = 1.f;
	static constexpr float InvMass = 1.f / Mass;
	glm::vec3 position, velocity;
	unsigned int lifeTime = 0;

	void Update() noexcept;

	AABB GetAABB() const noexcept
	{
		return AABB::MakeAABB(this->position + this->velocity * Tick::TimeDelta, this->position - Tick::TimeDelta);
	}
};
#endif // BULLET_H