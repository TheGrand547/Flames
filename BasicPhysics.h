#pragma once
#ifndef BASIC_PHYSICS_H
#define BASIC_PHYSICS_H
#include "glmHelp.h"
#include "util.h"

struct BasicPhysics
{
	glm::vec3 position{}, velocity{};
protected:
	float invMass = 1;
public:
	BasicPhysics() noexcept = default;
	BasicPhysics(const BasicPhysics& other) noexcept = default;
	BasicPhysics(BasicPhysics&& other) noexcept = default;
	~BasicPhysics() noexcept = default;
	BasicPhysics& operator=(const BasicPhysics& other) noexcept = default;

	inline BasicPhysics(const glm::vec3& position, const float& mass = 1) noexcept : position(position), velocity(), invMass(1.f / mass) {}
	inline BasicPhysics(const glm::vec3& position, const glm::vec3& velocity, const float& mass = 1) noexcept
		: position(position), velocity(velocity), invMass(1.f / mass) {}

	inline glm::vec3 ApplyForces(const glm::vec3& forces, const float& timeStep = Tick::TimeDelta) noexcept
	{
		// TODO: Set max speed somewhere
		// InvMass * timeStep will be a constant so it could be factored out but idk
		this->velocity += forces * this->invMass * timeStep;
		this->position += this->velocity * timeStep;
		return this->position;
	}

	inline void SetMass(const float& mass) noexcept
	{
		this->invMass = 1.f / mass;
	}

	static void Update(glm::vec3& position, glm::vec3& velocity, float mass, const glm::vec3& forces) noexcept;
	static void Update(glm::vec3& position, glm::vec3& velocity, const glm::vec3& forces, float mass = 1.f) noexcept;
};


#endif // BASIC_PHYSICS_H