#include "Player.h"
#include <numbers>

static constexpr float TurningModifier = Tick::TimeDelta * std::numbers::pi_v<float>;

void Player::Update(Input::Keyboard input) noexcept
{
	// Input is a 2d vector, x is thrust/deceleration, y is turning

	glm::vec3 unitVector = glm::normalize(this->velocity);

	glm::mat3 stored(static_cast<glm::mat3>(this->transform.rotation));

	glm::quat delta = glm::normalize(glm::angleAxis(TurningModifier * input.heading.y, glm::normalize(stored[1])));
	//glm::quat delta = glm::normalize(glm::angleAxis(TurningModifier * 0.5f, glm::vec3(0.f, 1.f, 0.f)));

	this->transform.rotation = delta * this->transform.rotation;
	//this->transform.Normalize();

	glm::vec3 forces = input.heading.x * stored[0];
	if (input.heading.x < 0 || glm::dot(this->velocity, forces) < 0.85f)
		forces *= 1.25f;

	forces *= 10.f;

	BasicPhysics::Update(this->transform.position, this->velocity, forces);
	if (glm::length(this->velocity) > 10.f)
	{
		this->velocity = glm::normalize(this->velocity) * 10.f;
	}
}
