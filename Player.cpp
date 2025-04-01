#include "Player.h"
#include <algorithm>
#include <numbers>
#include <glm/gtx/orthonormalize.hpp>
#include "Level.h"
#include "Animation.h"

static constexpr float TurningModifier = Tick::TimeDelta * std::numbers::pi_v<float>;

// This will be overhauled due to player mass increasing when "full"
static constexpr float PlayerMass = 5.f; // Idk

// TODO: Non "eyeball" tune these
static constexpr float MaxSpeed = 20.f;
static constexpr float minTurningRadius = 4.f;
static constexpr float EngineThrust = 40.f;
static constexpr float TurningThrust = 30.f;
static constexpr float MinFiringVelocity = 8.f; 

static constexpr IntervalType FireDelay = 16;
static constexpr IntervalType FireInterval = 200;
static constexpr IntervalType WaitingValue = -1;


static  constexpr float PlayerScale = 0.5f; // HACK
// Popcorn constants
static constexpr float PopcornSpeed = 100.f;


static const glm::vec3 PopcornOffset  = glm::vec3(3.5f, 0.f,  1.25f);
static const glm::vec3 PopcornOffsetZ = glm::vec3(3.5f, 0.f, -1.25f); // Painfully sloppy

// SPONSON OFFSET
static const glm::vec3 SponsonOffset = glm::vec3(6.75f, 0.f, 3.4f);
static const glm::vec3 SponsonOffsetZ = glm::vec3(6.75f, 0.f, -3.4f); // Painfully sloppy

static constexpr IntervalType PopcornDelay = 40;
static constexpr IntervalType RecoilTime = 15;
static SimpleAnimation popcornAnimation{ {glm::vec3(0.f)},  RecoilTime, Easing::EaseOutCircular,
						{glm::vec3(-0.4f, 0.f, 0.f)}, PopcornDelay - RecoilTime, Easing::EaseOutBack };

static AnimationInstance gunA, gunB;

glm::vec3 gunAPos, gunBPos; // You guessed it -- a hack

// TODO: Button to switch between simultaneous and alternating fire
// TODO: Compute shader for creating vertex positions of the hexagon based on a center and axes

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

	//const float angularVelocity = Tick::TimeDelta * Rectify(glm::sqrt(rotationalThrust / turningRadius));
	const float angularVelocity = Tick::TimeDelta * glm::pi<float>() * 0.75f;

	// Angular velocity is independent of mass
	rotationalThrust *= PlayerMass;

	// Input.heading.x is always positive, as it indicated the desired fraction
	const float desiredSpeed = input.heading.x * MaxSpeed;
	const float speedDifference = Rectify(glm::abs((desiredSpeed - currentSpeed) / currentSpeed));

	// Popcorn weapon firing
	if (input.popcornFire && (gunA.IsFinished() || gunB.IsFinished()))
	{
		// Popcorn fire
		glm::vec3 facingVector = localAxes[0];
		glm::vec3 bulletVelocity = facingVector * PopcornSpeed + this->velocity;
		//bulletVelocity = glm::normalize(bulletVelocity + this->velocity) * PopcornSpeed;
		if (gunA.IsFinished())
		{
			Level::AddBulletTree(this->transform.position + localAxes * (PopcornOffset * PlayerScale), bulletVelocity, localAxes[1]);
			popcornAnimation.Start(gunA);
		}
		if (gunB.IsFinished())
		{
			Level::AddBulletTree(this->transform.position + localAxes * (PopcornOffsetZ * PlayerScale), bulletVelocity, localAxes[1]);
			popcornAnimation.Start(gunB);
		}
	}
	gunAPos = popcornAnimation.Get(gunA).position;
	gunBPos = popcornAnimation.Get(gunB).position;

	glm::vec3 forces{0.f};

	// This is kind of a complete mess but it appears to work
	if (input.cruiseControl && this->sat)
	{
		// Entirely separate logic for "cruise control"
		//const glm::vec3 target = this->sat->GetBounding().GetCenter();
		const glm::vec3 target = Level::GetInterest();
		const glm::vec3 delta = glm::normalize(target - this->transform.position);
		glm::mat3 results{ 1.f };
		results[0] = delta;
		results[2] = glm::normalize(glm::cross(delta, localAxes[1]));
		results[1] = glm::normalize(glm::cross(results[2], results[0]));
		glm::quat transformation = glm::normalize(glm::quat(results));
		
		glm::quat change = glm::inverse(this->transform.rotation) * transformation;
		glm::vec3 axis = glm::normalize(glm::axis(change));

		float angleDot = glm::dot(transformation, this->transform.rotation);
		const float maxAngleChange = angularVelocity;

		float amount = glm::clamp(glm::angle(change), -maxAngleChange, maxAngleChange);

		bool significantDeviation = glm::epsilonNotEqual(amount, 0.f, glm::epsilon<float>()) &&
			glm::epsilonNotEqual(glm::abs(angleDot), 1.f, EPSILON);

		if (!glm::any(glm::isnan(axis)))
		{
			if (significantDeviation)
			{
				this->transform.rotation = glm::normalize(glm::rotate(this->transform.rotation, amount * glm::sign(angleDot), axis));
			}
			else
			{
				// This might've fixed the stuttering but I have no clue at this point man
				this->transform.rotation = transformation;
			}
		}
		glm::vec3 newForward = glm::rotate(this->transform.rotation, glm::vec3(1.f, 0.f, 0.f));
		glm::vec3 acceleration = glm::normalize(newForward - localAxes[0]);
		if (!glm::any(glm::isnan(acceleration)))
		{
			forces += acceleration * rotationalThrust;
		}
		else if (speedDifference < 1.f)
		{
			forces += newForward * input.heading.x * EngineThrust;
		}
		
		{
			glm::vec3 wrongDirection = glm::normalize(newForward - unitVector);
			float leng = glm::length(newForward - unitVector);
			if (leng < EPSILON)
			{
				this->velocity = newForward * currentSpeed;
			}
			else if (!glm::any(glm::isnan(wrongDirection)))
			{
				// Damping coeficient
				//forces += wrongDirection * input.heading.x * EngineThrust * leng;
				forces += newForward * input.heading.x * EngineThrust;
			}
			if (currentSpeed > desiredSpeed)
			{
				forces += unitVector * -EngineThrust * speedDifference;
			}
			this->velocity = newForward * currentSpeed;
		}
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
	else if (!this->fireCountdown)
	{
		glm::vec3 wrongDirection = glm::normalize(localAxes[0] - unitVector);
		if (!glm::any(glm::isnan(wrongDirection)))
		{
			// Damping coeficient
			forces += wrongDirection * input.heading.x * EngineThrust * 0.25f;
		}
	}
	
	// Do not allow for turning unless the ship is moving
	if (!this->fireCountdown)
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

		//Level::AddBullet(this->transform.position, bulletVelocity);
		this->velocity = -5.f * facingVector;
	}
	
	//forces = glm::vec3(0.f);
	//this->velocity = localAxes[0] * input.heading.x * MaxSpeed;

	// Move towards 6DOF
	forces = glm::vec3(0.f);
	glm::vec3 signFlags{ input.movement };
	// If a direction is not held, counteract the current force in that direction
	if (!glm::any(glm::isnan(unitVector)))
	{
		if (glm::abs(signFlags.x) < EPSILON)
		{
			signFlags.x = -1.f * (glm::dot(localAxes[0], unitVector)) * 0.25f;
		}
		if (glm::abs(signFlags.y) < EPSILON)
		{
			signFlags.y = -1.f * (glm::dot(localAxes[1], unitVector)) * 0.25f;
		}
		if (glm::abs(signFlags.z) < EPSILON)
		{
			signFlags.z = -1.f * (glm::dot(localAxes[2], unitVector)) * 0.25f;
		}
	}
	input.heading.x = 0.75f; // ?????
	forces += signFlags.x * localAxes[0] * input.heading.x * EngineThrust * 5.f;
	forces += signFlags.y * localAxes[1] * input.heading.x * EngineThrust * 5.f;
	forces += signFlags.z * localAxes[2] * input.heading.x * EngineThrust * 5.f;

	BasicPhysics::Update(this->transform.position, this->velocity, forces, PlayerMass);
	BasicPhysics::Clamp(this->velocity, MaxSpeed);
	if (glm::length(this->velocity) * Tick::TimeDelta < EPSILON)
	{
		this->velocity = glm::vec3(0.f);
	}
}

void Player::Draw(Shader& shader, VAO& vertex, MeshData& renderData, Model localModel) const noexcept
{
	shader.SetActiveShader();
	vertex.Bind();
	vertex.BindArrayBuffer(renderData.vertex);
	renderData.index.BindBuffer();
	// Render the static(non-animation bound) things
	localModel.scale = glm::vec3(PlayerScale);
	const glm::mat4 stored = localModel.GetModelMatrix();
	shader.SetMat4("modelMat", stored);
	shader.SetMat4("normalMat", localModel.GetNormalMatrix());
	shader.MultiDrawElements<DrawType::Triangle>(renderData.indirect, 0, 3);

	Model temp(gunAPos);
	shader.SetMat4("modelMat", stored * temp.GetModelMatrix());
	shader.DrawElements<DrawType::Triangle>(renderData.indirect, 3);
	temp.translation = gunBPos;
	shader.SetMat4("modelMat", stored * temp.GetModelMatrix());
	shader.DrawElements<DrawType::Triangle>(renderData.indirect, 4);
}
