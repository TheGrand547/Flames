#include "DemoGuy.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/random.hpp>
#include "Pathfinding.h"


DemoGuy::DemoGuy(glm::vec3 pos) noexcept : PathFollower(pos, 10.f), transform(pos, 10.f), stateCounter(0), lastFacing(1, 0, 0), currentState(States::Stare)
{
}

DemoGuy::~DemoGuy() noexcept
{
}

Model DemoGuy::GetModel() const noexcept
{
	Model result{ this->physics.position };
	return ;
}

Model DemoGuy::GetFacingModel() const noexcept
{
	return Model();
}

glm::mat4 DemoGuy::GetMod() const noexcept
{
	return glm::lookAt(glm::vec3(0.f), this->lastFacing, glm::vec3(0, 1, 0));
}

void DemoGuy::Update(glm::vec3 position) noexcept
{
	States nextState = States::Error;
	glm::vec3 forces{ 0 };
	switch (this->currentState)
	{
	case States::Track:
		break;
	case States::Stare:
	{

		break;
	}
	case States::Shoot:
	{
		Log("Should be unused");
		nextState = States::Stare;
		break;
	}
	case States::Transit:
	{
		this->PathUpdate();
		// BAD, grr
		this->transform = this->PathFollower::physics;
		if (this->path.size() == 0)
		{
			lastFacing = glm::normalize(this->PathFollower::physics.velocity);
			nextState = States::SlowDown;
		}
		break;
	}
	case States::SlowDown:
	{

		if (glm::length(this->transform.velocity) < EPSILON)
		{
			nextState = (rand() % 2 == 0) ? States::SlowDown : States::SlowDown;
		}
		break;
	}
	case States::Error:
		break;
	default:
		break;
	}
	// TODO: Collision detection and stuff
	this->PathFollower::physics.ApplyForces(forces, Tick::TimeDelta);
	// Too bad I'm ignoring that muahahahah

	if (nextState != States::Error)
	{
		switch (nextState)
		{
		case States::Track:
			break;
		case States::Transit:
		{
				// TODO: FIX BRUTAL HACK
			PathNodePtr start = nullptr, end = nullptr;
			glm::vec3 center = this->capsule.GetCenter();
			float minDist = INFINITY, minDist2 = INFINITY;
			for (auto& possible : PathNodes)
			{
				if (glm::distance(center, possible->GetPosition()) < minDist)
				{
					start = possible;
					minDist = glm::distance(center, possible->GetPosition());
				}
				if (glm::distance(position, possible->GetPosition()) < minDist)
				{
					end = possible;
					minDist2 = glm::distance(position, possible->GetPosition());
				}
			}
			if (start && end)
			{
				this->path = AStarSearch<PathNode>(start, end,
					[](const PathNode& a, const PathNode& b)
					{
						return glm::distance(a.GetPosition(), b.GetPosition());
					}
				);
			}
			break;
		}
		case States::SlowDown:
			break;
		case States::Shoot:
			break;
		case States::Stare:
		{
			// Wait between .5 and 4.5 seconds
			this->stateCounter = 64 + rand() % 128 * 4;
			this->lastFacing = glm::sphericalRand(1.f);
			break;
		}
		case States::Error:
			break;
		default:
			break;
		}
		this->currentState = nextState;
	}
}
