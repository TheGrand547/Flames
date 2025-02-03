#pragma once
#ifndef PLAYER_CONTROLLER_H
#define PLAYER_CONTROLLER_H
#include "glmHelp.h"
#include "Transform.h"
#include "BasicPhysics.h"
#include "Input.h"
#include "Model.h"

typedef std::uint8_t IntervalType;

class Player
{
protected:
	Transform transform;
	glm::vec3 velocity;
	// TODO: Probably move away from size_t's if possible
	IntervalType fireDelay, fireCountdown;

public:
	inline Player(const glm::vec3& position = glm::vec3(0), const glm::vec3& velocity = glm::vec3(0)) noexcept 
		: transform(position, glm::quat(glm::vec3(0.25f))), velocity(velocity), fireDelay(0), fireCountdown(0) {}

	// TODO: Encapsulate drawing functions

	void Update(Input::Keyboard input) noexcept;

	inline Model GetModel() noexcept
	{
		return { this->transform };
	}

	inline glm::vec3 GetVelocity() const noexcept
	{
		return this->velocity;
	}
};


#endif // PLAYER_CONTROLLER_H
