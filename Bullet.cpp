#include "Bullet.h"
#include "Level.h"

OBB Bullet::Collision{};

Bullet::Bullet(glm::vec3 position, glm::vec3 velocity, glm::vec3 up, unsigned int team) noexcept : transform(position, 
	ForwardDir(velocity, up)), speed(glm::length(velocity)), team(team)
{
}

void Bullet::Update() noexcept
{
	BasicPhysics::UpdateLinear(this->transform.position, this->transform.rotation * glm::vec3(this->speed, 0.f, 0.f));
	this->lifeTime++;
}
