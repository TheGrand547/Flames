#pragma once
#ifndef DEMO_GUY_H
#define DEMO_GUY_H
#include "BasicPhysics.h"
#include "glmHelp.h"
#include "Capsule.h"

#include "PathFollower.h"

enum class States
{
	// Looks around until eye contact with the player is made
	// Leads to Shoot Node
	Track, 
	
	// Going to the node where the last player kind of was
	// Leads to Slow Down
	Transit, 
	
	// Slows the guy down
	// Leads to Track or Stare
	SlowDown,

	// Shoots a ball at the player
	// Leads to Transit Node
	Shoot, 
	
	// Sits and stares in a random direction for a random amount of time
	// Leads to Track Node
	Stare,

	// Something went wrong
	Error
};

constexpr std::string GetPrintable(const States state) noexcept
{
	switch (state)
	{
	case States::Track: return "States::Track";
	case States::Transit: return "States::Transit";
	case States::SlowDown: return "States::SlowDown";
	case States::Stare: return "States::Stare";
	case States::Shoot: return "States::Shoot";
	case States::Error: return "States::Error";
	default: return "ERROR";
	}
}

class DemoGuy : public PathFollower
{
protected:
	BasicPhysics transform;
	States currentState = States::Error;
	std::uint16_t stateCounter;
	glm::vec3 lastFacing;
public:
	DemoGuy(glm::vec3 pos) noexcept;
	~DemoGuy() noexcept;

	Model GetModel() const noexcept;
	Model GetFacingModel() const noexcept;

	inline glm::vec3 GetFacing() const noexcept { return this->lastFacing; }
	glm::mat4 GetMod() const noexcept;

	// Advance one tick
	void Update(glm::vec3 position) noexcept;
};

#endif // DEMO_GUY_H