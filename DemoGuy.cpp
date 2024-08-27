#include "DemoGuy.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/random.hpp>
#include "Pathfinding.h"


DemoGuy::DemoGuy(glm::vec3 pos) noexcept : PathFollower(pos, 10.f), transform(pos, 1.f), stateCounter(0), lastFacing(1, 0, 0), currentState(States::Stare)
{
}

DemoGuy::~DemoGuy() noexcept
{
}

Model DemoGuy::GetModel() const noexcept
{
	Model result{ this->physics.position };
	return result;
}

Model DemoGuy::GetFacingModel() const noexcept
{
	return Model();
}

glm::mat4 DemoGuy::GetMod() const noexcept
{
	glm::mat4 looker{ SetForward(this->lastFacing) };
	looker[3] = glm::vec4(this->PathFollower::physics.position, 1.f);
	/*
	looker[0] *= 0.5f;
	looker[1] *= 0.75f;
	looker[2] *= 0.25f;
	*/
	return looker;
}

void DemoGuy::Update(glm::vec3 position) noexcept
{
	States nextState = States::Error;
	glm::vec3 forces{ 0 };
	switch (this->currentState)
	{
	case States::Track:
	{
		//Log("TODO: Track mode");
		glm::vec3 myPos = this->PathFollower::physics.position;
		glm::vec3 delta = glm::normalize(position - myPos);
		if (glm::abs(glm::dot(delta, this->lastFacing)) < glm::radians(35.f))
		{
			Log("SPOTTED");
			nextState = States::Transit;
		}
		
		//nextState = States::Stare;
		break;
	}
	case States::Stare:
	{
		if (this->stateCounter-- == 0)
		{
			nextState = States::Track;
			// Previous line is correct behavior, below is just demo
			//nextState = States::Stare;
		}
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
		//this->transform = this->PathFollower::physics;
		if (this->PathFollower::path.size() != 0)
		{
			this->lastFacing = glm::normalize(this->PathFollower::path.back()->GetPosition() - this->PathFollower::physics.position);
		}
		else
		{
			this->lastFacing = glm::normalize(this->PathFollower::physics.velocity);
			nextState = States::SlowDown;
		}
		break;
	}
	case States::SlowDown:
	{
		this->PathFollower::physics.ApplyForces(-glm::normalize(this->PathFollower::physics.velocity), Tick::TimeDelta);
		if (glm::length(this->transform.velocity) < EPSILON)
		{
			nextState = (rand() % 2 == 0) ? States::Stare : States::Stare;
		}
		break;
	}
	case States::Error:
		break;
	default:
		break;
	}
	// TODO: Collision detection and stuff
	//this->PathFollower::physics.ApplyForces(forces, Tick::TimeDelta);
	// Too bad I'm ignoring that muahahahah

	if (nextState != States::Error)
	{
		LogF("Heading to '%s' from '%s'\n", GetPrintable(nextState).c_str(), GetPrintable(this->currentState).c_str());
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
				if (glm::distance(position, possible->GetPosition()) < minDist2)
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
			// Wait between .5 and 1.5 seconds
			this->stateCounter = 64 + rand() % 128;
			this->lastFacing = glm::sphericalRand(1.f);
			Log("New direction: " << this->lastFacing);
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
