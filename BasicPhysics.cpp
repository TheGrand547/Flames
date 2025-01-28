#include "BasicPhysics.h"

void BasicPhysics::Update(glm::vec3& position, glm::vec3& velocity, float mass, const glm::vec3& forces) noexcept
{
	float invMass = 1.f / mass;
	velocity += forces * invMass * Tick::TimeDelta;
	position += velocity * Tick::TimeDelta;
}

void BasicPhysics::Update(glm::vec3& position, glm::vec3& velocity, const glm::vec3& forces, float mass) noexcept
{
	BasicPhysics::Update(position, velocity, mass, forces);
}

void BasicPhysics::Clamp(glm::vec3& velocity, const float& magnitude) noexcept
{
	float length = glm::length(velocity);
	if (length > magnitude)
	{
		velocity = glm::normalize(velocity) * magnitude;
	}
}
