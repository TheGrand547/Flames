#pragma once
#ifndef DOOR_H
#define DOOR_H
#include "glmHelp.h"
#include "Triangle.h"
#include "Sphere.h"
#include "Model.h"

// TODO: BSP on doors for navigation 
struct Door
{
	enum State
	{
		Opening, Closing, Closed, Open
	} openState;
	enum Status
	{
		Locked, Unlocked
	} lockedState;
	glm::vec3 position;
	Model model{}; // TODO: Proper construction

	enum class Type
	{
		Triangle, Square
	} openStyle = Type::Triangle;

	// If openTicks >= closingDuration then the door is closed
	int openTicks = 0;
	int closingDuration = 256;

	void StartOpening() noexcept;

	Door(glm::vec3 position, State state = Open, Status status = Unlocked) noexcept;

	void Update() noexcept;
	void Draw() noexcept;
	void Setup();

	std::array<Triangle, 2> GetTris() const noexcept;
	Sphere GetBroad() const noexcept;
};


#endif // DOOR_H
