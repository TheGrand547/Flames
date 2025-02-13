#include "Player.h"
#include <algorithm>
#include <numbers>
#include <glm/gtx/orthonormalize.hpp>
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

	glm::vec3 wrongDirection = glm::normalize(unitVector - localAxes[0]);
	if (!glm::any(glm::isnan(wrongDirection)))
	{
		//forces += wrongDirection * input.heading.x * EngineThrust;
	}

	if (input.cruiseControl && this->sat)
	{
		// Entirely separate logic for "cruise control"
		const glm::vec3 target = this->sat->GetBounding().GetCenter();
		const glm::vec3 delta = glm::normalize(target - this->transform.position);
		glm::mat3 results{ 1.f };
		results[0] = delta;
		results[2] = glm::normalize(glm::cross(delta, localAxes[1]));
		results[1] = glm::normalize(glm::cross(results[2], results[0]));
		glm::quat transformation = glm::normalize(glm::quat(results));
		
		glm::quat change = glm::inverse(this->transform.rotation) * transformation;
		glm::vec3 axis = glm::normalize(glm::axis(change));

		float angleDot = glm::dot(transformation, this->transform.rotation);
		const float maxAngleChange = Rectify(angularVelocity / glm::acos(angleDot));
		const float maxAngleChange2 = Rectify(angularVelocity);
		float amount = glm::clamp(glm::angle(change), -maxAngleChange2, maxAngleChange2);

		if (maxAngleChange != 0.f && glm::isfinite(maxAngleChange) && !glm::any(glm::isnan(axis)))
		{
			//this->transform.rotation = glm::slerp(this->transform.rotation, transformation, glm::abs(maxAngleChange));
			float b = glm::dot(this->transform.rotation, transformation);
			std::cout << "B:" << b;
			this->transform.rotation = this->transform.rotation * glm::normalize(glm::angleAxis(amount * glm::sign(b), axis));
			float a = glm::dot(this->transform.rotation, transformation);
			std::cout << "\tA:" << a << "\tD:" << maxAngleChange2 << "\tAbs:" << glm::abs(b - a);
			std::cout << "\tAV:" << angularVelocity << "\tV:" << currentSpeed << '\n';
		}
		// TODO: replace these static casts with simple rotates
		glm::mat3 updated = static_cast<glm::mat3>(this->transform.rotation);
		glm::vec3 acceleration = glm::normalize(updated[0] - localAxes[0]);

		// Probably the best working version, for no good reason I can tell
		//glm::vec3 acceleration = glm::normalize(delta - unitVector * glm::dot(delta, unitVector));

		//glm::vec3 acceleration = glm::normalize(updated[0] - localAxes[0] * glm::dot(updated[0], localAxes[0]));
		
		//glm::vec3 acceleration = glm::normalize(delta - localAxes[0] * glm::dot(delta, localAxes[0]));
		//glm::vec3 acceleration = glm::normalize(delta - updated[0] * glm::dot(delta, updated[0]));
		
		//acceleration = acceleration * glm::sign(glm::dot(acceleration, static_cast<glm::mat3>());
		//acceleration = glm::normalize(acceleration - localAxes[1] * glm::dot(localAxes[1], acceleration));
		//if (glm::abs(glm::dot(delta, unitVector)) < 0.88f)
		//{
		if (!glm::any(glm::isnan(acceleration)))
		{
			// This should not work
			//float hack = glm::sign(-axis[1]);
			forces += acceleration * rotationalThrust;
		}
		else if (speedDifference < 1.f)
		{
			forces += localAxes[0] * rotationalThrust;
		}
		//}
		BasicPhysics::Update(this->transform.position, this->velocity, forces, PlayerMass);
		BasicPhysics::Clamp(this->velocity, MaxSpeed);
		return;
	}
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
			//this->transform.rotation = glm::rotate(this->transform.rotation, angularVelocity * input.heading.y, localAxes[1]);
			forces -= input.heading.y * localAxes[2] * rotationalThrust;
		}
		if (input.heading.z != 0.f)
		{
			glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.z, glm::normalize(localAxes[2])));
			this->transform.rotation = delta * this->transform.rotation;
			//this->transform.rotation = glm::rotate(this->transform.rotation, angularVelocity * input.heading.z, localAxes[2]);
			forces += input.heading.z * localAxes[1] * rotationalThrust;
		}
		if (input.heading.w != 0.f)
		{
			glm::quat delta = glm::normalize(glm::angleAxis(angularVelocity * input.heading.w, glm::normalize(localAxes[0])));
			this->transform.rotation = delta * this->transform.rotation;
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
