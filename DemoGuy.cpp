#include "DemoGuy.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/random.hpp>
#include "Pathfinding.h"
#include "Level.h"


constexpr float slowTurnSpeed = 30.f;
constexpr float fastTurnSpeed = 60.f;


static const glm::mat3 slowTurnPos = glm::eulerAngleY(glm::radians(slowTurnSpeed / 128));
static const glm::mat3 slowTurnNeg = glm::eulerAngleY(-glm::radians(slowTurnSpeed / 128));
static const glm::mat3 fastTurnPos = glm::eulerAngleY(glm::radians(fastTurnSpeed / 128));
static const glm::mat3 fastTurnNeg = glm::eulerAngleY(-glm::radians(fastTurnSpeed / 128));

static constexpr float turnPeriodSeconds = 4.f;
static constexpr int turnPeriod = static_cast<int>(turnPeriodSeconds * 128);
static constexpr int turnPeriodHalf = turnPeriod / 2;
static constexpr int turnPeriodQuarter = turnPeriod / 4;
static constexpr int turnPeriodThreeFourths = 3 * turnPeriod / 4;


DemoGuy::DemoGuy(glm::vec3 pos) noexcept : PathFollower(pos, 1.f), transform(pos, 1.f), stateCounter(0), lastFacing(1, 0, 0), currentState(States::Stare)
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
		glm::vec3 myPos = this->PathFollower::physics.position;
		glm::vec3 delta = glm::normalize(position - myPos);
		if (glm::abs(glm::acos(glm::dot(delta, this->lastFacing))) < glm::radians(35.f))
		{
			Log("SPOTTED");
			nextState = States::Transit;
		}
		else
		{
			this->stateCounter++;
			std::uint16_t modulo = this->stateCounter % turnPeriod;
			this->lastFacing = ((modulo < turnPeriodQuarter || modulo > turnPeriodThreeFourths) ? slowTurnPos : slowTurnNeg )* this->lastFacing;
		}
		break;
	}
	case States::Stare:
	{
		if (this->stateCounter-- == 0)
		{
			nextState = States::Track;
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
		this->stateCounter++;
		this->PathFollower::physics.ApplyForces(-glm::normalize(this->PathFollower::physics.velocity), Tick::TimeDelta);
		if (glm::length(this->transform.velocity) < EPSILON)
		{
			Log("Slowdown Frames: " << this->stateCounter);
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
		LogF("Heading from '%s' to '%s'\n", GetPrintable(this->currentState).c_str(), GetPrintable(nextState).c_str());
		switch (nextState)
		{
		case States::Track:
		{
			this->stateCounter = turnPeriodHalf;
			break;
		}
		case States::Transit:
		{
			PathNodePtr start = nullptr, end = nullptr;
			glm::vec3 center = this->capsule.GetCenter();
			start = Level::Tree.nearestNeighbor(center);
			end = Level::Tree.nearestNeighbor(position);
			// TODO: If this doesn't crash for long enough remove it
#ifdef DEBUG
			float minDist = INFINITY, minDist2 = INFINITY;
			PathNodePtr start2 = nullptr, end2 = nullptr;
			for (auto& possible : Level::AllNodes)
			{
				if (glm::distance(center, possible->GetPosition()) < minDist)
				{
					start2 = possible;
					minDist = glm::distance(center, possible->GetPosition());
				}
				if (glm::distance(position, possible->GetPosition()) < minDist2)
				{
					end2 = possible;
					minDist2 = glm::distance(position, possible->GetPosition());
				}
			}
			if (start != start2 || end != end2)
			{
				Log("Failure to line up.");
				std::cout << glm::distance(start->GetPos(), center) << ":" << minDist << '\n';
				std::cout << glm::distance(end->GetPos(), center) << ":" << minDist2 << '\n';
			}
#endif // DEBUG
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
		{
			this->stateCounter = 0;
			break;
		}
		case States::Shoot:
			break;
		case States::Stare:
		{
			// Wait between .5 and 1.5 seconds
			this->stateCounter = 64 + rand() % 128;
			this->lastFacing = circleRand(1.f);
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
