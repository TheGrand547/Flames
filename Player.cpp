#include "Player.h"
#include <algorithm>
#include <numbers>

static constexpr float TurningModifier = Tick::TimeDelta * std::numbers::pi_v<float>;
static constexpr float PlayerMass = 5.f; // Idk
static constexpr float MaxSpeed = 20.f;

void Player::Update(Input::Keyboard input) noexcept
{
	// Input is a 2d vector, x is thrust/deceleration, y is turning

	constexpr float EngineThrust = 20.f;
	constexpr float TurningThrust = 20.f;

	const glm::vec3 unitVector = glm::normalize(this->velocity);

	const glm::mat3 stored(static_cast<glm::mat3>(this->transform.rotation));

	const float rotationSpeed = glm::length(this->velocity);
	const float v2 = rotationSpeed * rotationSpeed;

	const float minTurningRadius = 10.f;

	// Calculate the correct thrust to apply a rotation, attempting to keep the speed of the ship constant
	float rotationalThrust = glm::min(TurningThrust, v2 / minTurningRadius);
	float turningRadius = glm::max(v2 / rotationalThrust, minTurningRadius);
	rotationalThrust = v2 / turningRadius;

	const float angularVelocity = Tick::TimeDelta * Rectify(glm::sqrt(rotationalThrust / turningRadius));

	// Angular velocity is independent of mass
	rotationalThrust *= PlayerMass;

	glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.y, glm::normalize(stored[1])));

	// TODO: Figure out how to handle mouse delta better
	glm::quat delta2 = glm::normalize(glm::angleAxis(angularVelocity * input.heading.z, glm::normalize(stored[2])));
	this->transform.rotation = delta * this->transform.rotation;
	this->transform.rotation = delta2 * this->transform.rotation;

	//glm::vec3 forces = input.heading.x * stored[0];
	glm::vec3 forces = stored[0] * EngineThrust;

	if (rotationSpeed > EPSILON)
	{
		if (input.heading.y != 0.f)
		{
			forces -= input.heading.y * stored[2] * rotationalThrust;
		}
		if (input.heading.z != 0.f)
		{
			forces += input.heading.z * stored[1] * rotationalThrust;
		}
	}

	//if (input.heading.x < 0 || glm::dot(this->velocity, forces) < 0.85f)
		//forces *= 1.25f;
	//forces += input.heading.y * stored[2] * 0.25f * (1.f - glm::dot(stored[2], glm::normalize(this->velocity)));

	BasicPhysics::Update(this->transform.position, this->velocity, forces, PlayerMass);
	if (glm::length(this->velocity) > MaxSpeed)
	{
		this->velocity = glm::normalize(this->velocity) * MaxSpeed;
	}
}
