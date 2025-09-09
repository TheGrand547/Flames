#pragma once
#ifndef BULLET_H
#define BULLET_H
#include "glmHelp.h"
#include "BasicPhysics.h"
#include "AABB.h"
#include "OrientedBoundingBox.h"

struct Bullet
{
	static constexpr float Mass = 1.f;
	static constexpr float InvMass = 1.f / Mass;
	
	static OBB Collision;
	
	//glm::vec3 position, velocity, up;
	Transform transform;
	float speed;
	unsigned int lifeTime = 0;
	unsigned int team = 0;

	void Update() noexcept;

	Bullet(glm::vec3 position, glm::vec3 velocity, glm::vec3 up, unsigned int team = 0) noexcept;
	
	Model GetModel() const noexcept;
	OBB GetOBB() const noexcept;
	AABB GetAABB() const noexcept;
	
	inline bool IsValid() const noexcept
	{
		return !glm::any(glm::isnan(this->transform.position)) && this->lifeTime < 5 * Tick::PerSecond;
	}
};
#endif // BULLET_H