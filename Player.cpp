#include "Player.h"
#include <algorithm>
#include <numbers>

static constexpr float TurningModifier = Tick::TimeDelta * std::numbers::pi_v<float>;
static constexpr float PlayerMass = 5.f; // Idk
static constexpr float MaxSpeed = 20.f;
static constexpr float minTurningRadius = 5.f;
static constexpr float EngineThrust = 40.f;
static constexpr float TurningThrust = 60.f;
static constexpr float DecayTicks = 128.f;
static constexpr float DecayAmount = 0.9f;
static float DecayConstant = std::exp(std::log(DecayAmount) / DecayTicks);

void Player::Update(Input::Keyboard input) noexcept
{
	// Input is a 3d vector, x is thrust/deceleration, y is horizontal turning, z is vertical turning
	const glm::vec3 unitVector = glm::normalize(this->velocity);

	const glm::mat3 stored(static_cast<glm::mat3>(this->transform.rotation));

	const float speed = glm::length(this->velocity);
	const float v2 = speed * speed;

	// Calculate the correct thrust to apply a rotation, attempting to keep the speed of the ship constant
	float rotationalThrust = glm::min(TurningThrust, v2 / minTurningRadius);
	float turningRadius = glm::max(v2 / rotationalThrust, minTurningRadius);
	rotationalThrust = v2 / turningRadius;

	const float angularVelocity = Tick::TimeDelta * Rectify(glm::sqrt(rotationalThrust / turningRadius));

	// Angular velocity is independent of mass
	rotationalThrust *= PlayerMass;

	// Input.heading.x is always positive, as it indicated the desired fraction
	const float desiredSpeed = input.heading.x * MaxSpeed;
	const float speedDifference = Rectify(glm::abs((desiredSpeed - speed) / speed));

	glm::vec3 forces{0.f};
	if (speedDifference > EPSILON)
	{
		if (speed < desiredSpeed)
		{
			forces = input.heading.x * stored[0] * EngineThrust;
		}
		else
		{
			forces = unitVector * -EngineThrust * (1.f - speedDifference);
		}
	}

	if (speed > EPSILON)
	{
		if (input.heading.y != 0.f)
		{
			glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.y, glm::normalize(stored[1])));
			this->transform.rotation = delta * this->transform.rotation;
			forces -= input.heading.y * stored[2] * rotationalThrust;
		}
		if (input.heading.z != 0.f)
		{
			glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.z, glm::normalize(stored[2])));
			this->transform.rotation = delta * this->transform.rotation;
			forces += input.heading.z * stored[1] * rotationalThrust;
		}
	}

	BasicPhysics::Update(this->transform.position, this->velocity, forces, PlayerMass);
	BasicPhysics::Clamp(this->velocity, MaxSpeed);
	if (glm::length(this->velocity) > (input.heading.x * MaxSpeed))
	{
		//this->velocity *= DecayConstant;
	}
}
