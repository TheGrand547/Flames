#include "ClockBrain.h"
#include <glm/gtc/random.hpp>
#include "Model.h"
#include "BasicPhysics.h"
#include "MissileMotion.h"
#include "log.h"

void ClockBrain::Init()
{
	this->transform.position = glm::ballRand(50.f);
	this->transform.rotation = glm::angleAxis(glm::gaussRand(glm::pi<float>(), glm::pi<float>()), glm::sphericalRand(1.f));
	this->velocity = this->transform.rotation * glm::vec3(1.f, 0, 0);
	this->target = this->transform.position;
	this->home = glm::i16vec3(this->target);
	this->state = 0;
}

void ClockBrain::Update()
{
	glm::vec3 thingVelocity{ 0.f };
	// STATE Thingy
	// state=0 default, patrol position      <- check player delta every 32 ticks
	// state=1 transit to somewhere          <- check player delta every  8 ticks
	// state=2 look for player where spotted <- check player delta every  8 ticks
	// state=3 return to home                <- check player delta every 32 ticks
	// state=4 I see the player

	// Ensure that not every single enemy does this check every frames
	const std::size_t modulatedTick = Level::GetCurrentTick() + (std::bit_cast<std::size_t>(this) >> 6) & 0b111111;
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
				//thingVelocity = Level::GetPlayerVel();
				this->target = playerPos;
				this->state = 1;
				if (modulatedTick % 256 == 0)
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
	glm::vec3 forced = MakePrediction(this->transform.position, this->velocity, 40.f, this->target, thingVelocity);
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

void ClockBrain::Draw(MeshData& data, VAO& vao, Shader& shader) const
{
	shader.SetActiveShader();
	vao.Bind();
	vao.BindArrayBuffer(data.vertex);
	data.index.BindBuffer();
	Model model{ this->transform };
	shader.SetMat4("modelMat", model.GetModelMatrix());
	shader.SetMat4("normalMat", model.GetNormalMatrix());
	shader.DrawElements(data.indirect);
}
