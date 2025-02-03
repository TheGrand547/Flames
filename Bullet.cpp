#include "Bullet.h"
#include "Level.h"

void Bullet::Update() noexcept
{
	// TODO: Forces
	glm::vec3 forces{ 0.f };

	BasicPhysics::Update(this->position, this->velocity, forces, Bullet::Mass);
}
