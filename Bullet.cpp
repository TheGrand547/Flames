#include "Bullet.h"
#include "Level.h"

OBB Bullet::Collision{};

Bullet::Bullet(glm::vec3 position, glm::vec3 velocity, glm::vec3 up, unsigned int team) noexcept : transform(position, 
	ForwardDir(velocity, up)), speed(glm::length(velocity)), team(team)
{
}

Model Bullet::GetModel() const noexcept
{
	return Model(this->transform);
}

OBB Bullet::GetOBB() const noexcept
{
	OBB copied = Bullet::Collision;
	copied.Rotate(this->GetModel().GetModelMatrix());
	return copied;
}

AABB Bullet::GetAABB() const noexcept
{
	return this->GetOBB().GetAABB();
}

void Bullet::Update() noexcept
{
	BasicPhysics::UpdateLinear(this->transform.position, this->transform.rotation * glm::vec3(this->speed, 0.f, 0.f));
	this->lifeTime++;
}
