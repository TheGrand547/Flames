#include "ClockBrain.h"
#include <glm/gtc/random.hpp>
#include "Model.h"
#include "BasicPhysics.h"
#include "MissileMotion.h"
#include "log.h"
#include "Interpolation.h"
#include "Geometry.h"
#include "BinarySpacePartition.h"
#include "ResourceBank.h"

static constexpr float InfluenceRadius = 15.f;
static constexpr float CohesionForce = 5.f;
static constexpr float AlignmentForce = 5.f;
static constexpr float SeparationForce = 20.f;
static constexpr float WanderForce = 8.f;
static constexpr float PlayerSightForce = 12.f;
// Since it's only once every three frames, triple the strength -- might look jerky but who cares
static constexpr float CollisionAvoidForce = 36.f;

static constexpr float ReturnForceMax = 50.f;
static constexpr float ReturnForceMinRadius = 20.f;
static constexpr float ReturnForceMaxRadius = 70.f;


static constexpr float MaxClockSpeed = 15.f;

glm::vec3 wandering(float& in, glm::vec3 forward)
{
	in += Tick::TimeDelta * glm::linearRand(-2 * glm::pi<float>(), 2 * glm::pi<float>());
	glm::vec3 randomPoint = glm::vec3(glm::cos(in), 0, glm::sin(in));
	glm::vec3 pointAhead = glm::normalize(forward * 5.f + randomPoint);
	return pointAhead * (WanderForce);
}

glm::vec3 drifer(glm::vec3 pos, glm::vec3 target)
{
	glm::vec3 forcing{};
	float dist = glm::length(target - pos);
	if (dist < 1)
	{
		return glm::vec3(0.f);
	}
	float constant = glm::pow(glm::max(0.f, (dist - ReturnForceMinRadius) / ReturnForceMaxRadius), 2.f);
	forcing = glm::normalize(target - pos) * constant * ReturnForceMax;
	return forcing;
}

bool ValidPOsition(const glm::vec3& vec)
{
	return Bank<BSP>::Get("Fellas").TestPoint(vec);
}

void ClockBrain::Init()
{

	do {
		this->transform.position = glm::ballRand(50.f);
	} while (!ValidPOsition(this->transform.position));
	do
	{
		this->home = glm::i16vec3(this->transform.position + glm::ballRand(10.f));
	} while (!ValidPOsition(this->home));
	this->transform.position = glm::vec3(0.f, 60.f, 0.f);

	this->transform.rotation = glm::angleAxis(glm::gaussRand(glm::pi<float>(), glm::pi<float>()), glm::sphericalRand(1.f));
	this->transform.rotation = ForwardDir(glm::vec3(0.f, -1.f, 0.f), glm::vec3(1.f, 0.f, 0.f));

	this->velocity = this->transform.rotation * glm::vec3(1.f, 0, 0);
	this->target = this->transform.position;
	this->home = glm::i16vec3(this->target + glm::ballRand(10.f));


	this->home = glm::i16vec3(this->target);
	this->state = 0;
	this->tickOffset = rand() % 256;
}

void ClockBrain::Update(const kdTree<Transform>& transforms)
{
	glm::vec3 thingVelocity{ 0.f };
	bool playerSpotted = false;
	// STATE Thingy
	// state=0 default, patrol position      <- check player delta every 32 ticks
	// state=1 transit to somewhere          <- check player delta every  8 ticks
	// state=2 look for player where spotted <- check player delta every  8 ticks
	// state=3 return to home                <- check player delta every 32 ticks
	// state=4 I see the player

	// Ensure that not every single enemy does this check every frames
	//const std::size_t modulatedTick = Level::GetCurrentTick() + (std::bit_cast<std::size_t>(this) >> 6) & 0b111111;
	const std::size_t modulatedTick = this->GetModulatedTick();
	//if (((this->state == 0 || this->state == 3) && modulatedTick % 32 == 0) || 
		//((this->state == 1 || this->state == 2) && modulatedTick % 8 == 0))
	{
		const glm::vec3 playerPos = Level::GetPlayerPos();
		const glm::vec3 difference = playerPos - this->transform.position;
		glm::mat3 current = glm::mat3_cast(this->transform.rotation);

		float threshold = glm::pi<float>() / 4.f;
		// Enemy is more alert if it isn't ambiently patrolling
		if (this->state != 0)
		{
			threshold *= 2.f;
		}
		if (glm::abs(glm::dot(current[0], glm::normalize(difference))) > glm::cos(threshold) &&
			glm::length(difference) < 40.f)
		{
			// Raycast
			Ray liota(this->transform.position, difference);
			RayCollision range{};
			bool clearSight = true;
			for (const auto& tri : Level::GetTriangleTree().RayCast(liota))
			{
				if (tri->RayCast(liota, range) && range.depth < glm::length(difference))
				{
					clearSight = false;
					break;
				}
			}
			if (clearSight)
			{
				playerSpotted = true;
				//thingVelocity = Level::GetPlayerVel();
				
				// TODO: REVERT, I am testing their avoidance of obstacles
				this->target = playerPos;
				this->state = 1;
				if (modulatedTick % 128 == 0)
				{
					Level::AddBulletTree(this->transform.position, current[0] * 80.f, 
						current[1], 1);
				}
			}
		}
	}
	if (glm::distance(this->transform.position, this->target) < 0.5)
	{
		// Generate new target
		//this->target = glm::vec3(this->home) + glm::sphericalRand(10.f);//glm::ballRand(100.f);
		switch (this->state)
		{
		case 0: // Standard Patrol
		{
			this->target = glm::vec3(this->home) + glm::sphericalRand(10.f);
			break;
		}
		case 1:
		{
			this->target += glm::ballRand(10.f);
			// TODO: Have counter to do this x number of times or something
			this->state = 2;
			break;
		}
		case 2:
		{
			this->state = 3;
			this->target = glm::vec3(this->home);
			break;
		}
		case 3:
		{
			this->target = glm::vec3(this->home) + glm::sphericalRand(10.f);
			this->state = 0;
			break;
		}
		default:
		{
			Log("State Error");
			break;
		}
		}
		if (glm::length(this->target) > 200.f)
		{
			std::cout << this->target << '\n';
			std::cout << this->home << ":" << glm::vec3(this->home) << '\n';
			this->target = glm::vec3(0.f);
		}
	}

	// Always more towards the target
	//glm::vec3 acceleration = MakePrediction(this->transform.position, this->velocity, 40.f, this->target, thingVelocity);
	glm::vec3 acceleration = MakePrediction(this->transform.position, this->velocity, 40.f, Level::GetPlayerPos(), thingVelocity);
	glm::vec3 flockingForces = this->IndirectUpdate(transforms);
	acceleration += drifer(this->transform.position, this->home);
	if (!playerSpotted)
	{
		acceleration += wandering(this->wander, this->transform.rotation * glm::vec3(1.f, 0.f, 0.f));
	}
	acceleration += flockingForces;
	// I'm not sure why it's one?
	if (playerSpotted) // Player is in sight
	{
		//acceleration += glm::normalize(Level::GetPlayerPos() - this->transform.position) * PlayerSightForce;
	}

	BasicPhysics::Update(this->transform.position, this->velocity, acceleration);
	BasicPhysics::Clamp(this->velocity, MaxClockSpeed);
	// Pretend it's non-zero
	if (glm::length(this->velocity) > EPSILON)
	{
		glm::mat3 orient{};
		glm::mat3 current = glm::mat3_cast(this->transform.rotation);
		orient[0] = glm::normalize(this->velocity);
		orient[1] = glm::cross(current[2], orient[0]);
		orient[2] = glm::cross(orient[0], orient[1]);
		this->transform.rotation = glm::normalize(glm::quat_cast(orient));
	}
	if (modulatedTick % 2 == 0)
	{
		/*
		OBB tight = this->GetOBB();
		const AABB broad = tight.GetAABB();
		for (const auto& tri : Level::GetTriangleTree().Search(broad))
		{
			::Collision range{};
			// TODO: Collision detection that actually gives you the overlap? Why didn't you do this beforehand
			if (DetectCollision::Overlap(tight, *tri, range))
			{
				float alignment = glm::dot(this->velocity, range.axis);
				this->velocity -= 2 * alignment * range.axis;
				this->transform.position += range.axis * range.depth;
			}
		}
		*/
	}
}

void ClockBrain::Update2(const kdTree<Transform>& transforms)
{
	glm::vec3 direction = this->IndirectUpdate(transforms);
	glm::vec3 forced = MakePrediction(this->transform.position, this->velocity, 10.f, glm::vec3(0.f), glm::vec3(0.f));
	forced = drifer(this->transform.position, this->home);
	forced += wandering(this->wander, this->transform.rotation * glm::vec3(1.f, 0.f, 0.f));
	forced += direction;
	BasicPhysics::Update(this->transform.position, this->velocity, forced);
	BasicPhysics::Clamp(this->velocity, 10.f);
	// Pretend it's non-zero
	if (glm::length(this->velocity) > EPSILON)
	{
		glm::mat3 orient{};
		glm::mat3 current = glm::mat3_cast(this->transform.rotation);
		orient[0] = glm::normalize(this->velocity);
		orient[1] = glm::cross(current[2], orient[0]);
		orient[2] = glm::cross(orient[0], orient[1]);
		this->transform.rotation = glm::normalize(glm::quat_cast(orient));
	}
}


glm::vec3 ClockBrain::IndirectUpdate(const kdTree<Transform>& transforms) noexcept
{
	std::vector<Transform> withinRadius = transforms.neighborsInRange(this->transform.position, InfluenceRadius);
	//std::cout << withinRadius.size() << '\n';
	glm::vec3 forward = this->transform.rotation * glm::vec3(1.f, 0.f, 0.f);
	glm::vec3 cohesion{}, separation{}, alignment{};
	glm::vec3 positionAverage{ }, forwardAverage{ }, separationTotal{};
	const float inverse = 1.f / static_cast<float>(withinRadius.size());
	for (const Transform& element : withinRadius)
	{
		//if (glm::length(this->transform.position - element.position) < 5.f)
		{
			glm::vec3 delta = this->transform.position - element.position;
			float scalar = glm::length(delta);
			if (scalar > EPSILON)
			{
				separationTotal += glm::normalize(delta) * SeparationForce / scalar;
			}
		}
		positionAverage += element.position * inverse;
		forwardAverage  += element.rotation * glm::vec3(1.f, 0.f, 0.f) * inverse;
	}
	// TODO: Collision avoidance factor
	// Ensure NAN doesn't propogate
	if (glm::length(positionAverage - this->transform.position) > EPSILON)
	{
		cohesion = glm::normalize(positionAverage - this->transform.position) * CohesionForce;
	}
	if (glm::length(forwardAverage - forward) > EPSILON)
	{
		alignment = glm::normalize(forwardAverage - forward) * AlignmentForce;
	}
	if (glm::length(separationTotal) > EPSILON)
	{
		float respulsion = glm::length(separationTotal);
		//std::cout << respulsion << '\n';
		if (respulsion < SeparationForce)
		{
			separation = glm::normalize(separationTotal) * SeparationForce;
		}
		else
		{
			separation = separationTotal;
		}
		separation = separationTotal;
	}
	glm::vec3 avoidance{};
	const std::size_t modulatedTick = this->GetModulatedTick();
	if ((modulatedTick & 3) == 3 || true)
	{
		/*
		glm::vec3 avoid{};
		OBB forward = this->GetOBB();
		glm::vec3 sizes = forward.GetScale();
		// Look for things up to two seconds ahead of our bloke
		constexpr float LookAheadTime = 1.5f;
		
		const float forwardLength = Rectify(glm::length(this->velocity));

		// So original forward is x, we want to scale it in place 
		// (sizes[0] + Rectify(length)) / sizes[0]
		float proper = (sizes[0] + forwardLength * LookAheadTime) / sizes[0];

		forward.Scale(glm::vec3(proper, 1.f, 1.f));
		forward.Translate(proper * sizes[0] * forward.Forward()); // So dang sketchy
		for (const auto& tri : Level::GetTriangleTree().Search(forward.GetAABB()))
		{
			::Collision range{};
			if (DetectCollision::Overlap(forward, *tri, range))
			{
				glm::vec3 tempy = tri->GetNormal();
				avoid += range.axis;
			}
		}
		if (Rectify(glm::length(avoid)) > EPSILON)
		{
			avoidance = glm::min(CollisionAvoidForce, CollisionAvoidForce) * glm::normalize(avoid);
		}
		*/
		glm::vec3 start = this->transform.position;
		glm::vec3 direction = glm::normalize(this->velocity);
		const OBB& target = Bank<OBB>::Get("NoGoZone");
		RayCollision rayman{};
		if (target.Intersect(start, direction, rayman) && rayman.distance < glm::length(this->velocity) * 1.5f)
		{
			avoidance = (CollisionAvoidForce * 1.5f) * rayman.normal;
		}
	}
	return alignment + cohesion + separation + avoidance;
}

void ClockBrain::Draw(MeshData& data, VAO& vao, Shader& shader) const
{
	shader.SetActiveShader();
	vao.Bind();
	vao.BindArrayBuffer(data.vertex);
	shader.SetVec3("shapeColor", glm::vec3(1.f));
	Model model{ this->transform };
	shader.SetMat4("modelMat", model.GetModelMatrix());
	shader.SetMat4("normalMat", model.GetNormalMatrix());
	shader.DrawElements<DrawType::Lines>(data.index);
	data.index.BindBuffer();
	//glDrawElements(GL_TRIANGLES, data.index.GetElementCount(), GL_UNSIGNED_INT, static_cast<const void*>(nullptr));
	model.translation += glm::vec3(0.f, 10.f, 0.f);
	shader.SetMat4("modelMat", model.GetModelMatrix());
	shader.SetMat4("normalMat", model.GetNormalMatrix());
	shader.DrawArray(data.vertex);
	//glDrawArrays(GL_TRIANGLES, 0, data.vertex.GetElementCount());
}
