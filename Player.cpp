#include "Player.h"
#include <algorithm>
#include <numbers>
#include "Level.h"

static constexpr float TurningModifier = Tick::TimeDelta * std::numbers::pi_v<float>;

// This will be overhauled due to player mass increasing when "full"
static constexpr float PlayerMass = 5.f; // Idk

// TODO: Non "eyeball" tune these
static constexpr float MaxSpeed = 20.f;
static constexpr float minTurningRadius = 4.f;
static constexpr float EngineThrust = 40.f;
static constexpr float TurningThrust = 40.f;
static constexpr float MinFiringVelocity = 8.f; 

static constexpr IntervalType FireDelay = 16;
static constexpr IntervalType FireInterval = 200;
static constexpr IntervalType WaitingValue = -1;

void Player::SelectTarget() noexcept
{
	float alignment = 0.f;
	std::size_t index = -1;
	const glm::mat3 localAxes(static_cast<glm::mat3>(this->transform.rotation));
	const glm::vec3 forward = localAxes[0];
	const glm::vec3 currentPos{};

	//for(...)
	{
		glm::vec3 otherPos = World::Zero;
		glm::vec3 delta = otherPos - currentPos;
		glm::vec3 normalized = glm::normalize(delta);

		float directionDelta = glm::dot(normalized, forward);

		// Arbitrary cutoff
		if (directionDelta > 0.5f && directionDelta < alignment)
		{
			alignment = directionDelta;
			index = 0;
		}

	}

}

void Player::Update(Input::Keyboard input) noexcept
{
	// Input is a 3d vector, x is thrust, y is horizontal turning, z is vertical turning
	const glm::vec3 unitVector = glm::normalize(this->velocity);

	const glm::mat3 localAxes(static_cast<glm::mat3>(this->transform.rotation));

	const float currentSpeed = glm::length(this->velocity);
	const float velocitySquared = currentSpeed * currentSpeed;

	// Relevant equation for circular motion; A = v*v/r = w*w*r, 
	// A is acceleration towards the center of rotation, v is tangential velocity, r is the radius, w is the angular velocity
	// Calculate the correct thrust to apply a rotation, attempting to keep the speed of the ship constant
	float rotationalThrust = glm::min(TurningThrust, velocitySquared / minTurningRadius);
	const float turningRadius = glm::max(velocitySquared / rotationalThrust, minTurningRadius);
	rotationalThrust = velocitySquared / turningRadius;

	const float angularVelocity = Tick::TimeDelta * Rectify(glm::sqrt(rotationalThrust / turningRadius));

	// Angular velocity is independent of mass
	rotationalThrust *= PlayerMass;

	// Input.heading.x is always positive, as it indicated the desired fraction
	const float desiredSpeed = input.heading.x * MaxSpeed;
	const float speedDifference = Rectify(glm::abs((desiredSpeed - currentSpeed) / currentSpeed));

	glm::vec3 forces{0.f};
	if (speedDifference > EPSILON)
	{
		if (currentSpeed < desiredSpeed)
		{
			forces = localAxes[0] * input.heading.x * EngineThrust;
		}
		else
		{
			// Stronger 'deceleration' when target speed is further from current speed
			forces = unitVector * -EngineThrust * speedDifference;
		}
	}
	
	// Do not allow for turning unless the ship is moving
	if (currentSpeed > EPSILON && !this->fireCountdown)
	{
		// Don't do extra quaternion math if we don't have to
		if (input.heading.y != 0.f)
		{
			glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.y, glm::normalize(localAxes[1])));
			this->transform.rotation = delta * this->transform.rotation;
			forces -= input.heading.y * localAxes[2] * rotationalThrust;
		}
		if (input.heading.z != 0.f)
		{
			glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.z, glm::normalize(localAxes[2])));
			this->transform.rotation = delta * this->transform.rotation;
			forces += input.heading.z * localAxes[1] * rotationalThrust;
		}
	}

	if (input.fireButton && this->fireDelay == 0 && this->fireCountdown == 0 && currentSpeed > MinFiringVelocity)
	{
		this->fireCountdown = FireDelay;
		this->fireDelay = WaitingValue;
	}
	if (this->fireDelay != WaitingValue && this->fireDelay)
	{
		this->fireDelay--;
	}

	if (this->fireCountdown)
	{
		this->fireCountdown--;
		forces = World::Zero;
	}
	// Time to fire
	else if (this->fireDelay == WaitingValue)
	{
		this->fireDelay = FireInterval;
		// TODO: See if kinetic energy "feels" better
		
		// TODO: There's a constant to factor out here
		// KE = 1/2 * m * v^2

		glm::vec3 facingVector = static_cast<glm::mat3>(this->transform.rotation)[0];
		facingVector = unitVector;
		float kineticEnergy = 0.5f * PlayerMass * currentSpeed * currentSpeed;
		float bulletEnergy = glm::sqrt(kineticEnergy * 2.f * Bullet::InvMass);

		float momentum = currentSpeed * PlayerMass;
		float bulletMomentum = momentum * Bullet::InvMass;
		// Conserve momentum
		
		glm::vec3 bulletVelocity = bulletEnergy * facingVector;

		Level::AddBullet(this->transform.position, bulletVelocity);
		this->velocity = -5.f * facingVector;
	}
	BasicPhysics::Update(this->transform.position, this->velocity, forces, PlayerMass);
	BasicPhysics::Clamp(this->velocity, MaxSpeed);
}
