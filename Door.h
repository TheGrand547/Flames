#pragma once
#ifndef DOOR_H
#define DOOR_H
#include "glmHelp.h"

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
	int openTicks = 0;
	int openTime = 256;


	Door(glm::vec3 position, State state = Open, Status status = Unlocked) noexcept;

	void Update() noexcept;
	void Draw() noexcept;
	void Setup();
};


#endif // DOOR_H
