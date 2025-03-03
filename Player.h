#pragma once
#ifndef PLAYER_CONTROLLER_H
#define PLAYER_CONTROLLER_H
#include "glmHelp.h"
#include "Transform.h"
#include "BasicPhysics.h"
#include "Input.h"
#include "Model.h"
#include "Satelite.h"
#include "VertexArray.h"

typedef std::uint8_t IntervalType;

class Player
{
protected:
	Transform transform;
	glm::vec3 velocity;
	IntervalType fireDelay, fireCountdown;

	void SelectTarget() noexcept;

public:
	Satelite* sat = nullptr; // AHHHHH

	inline Player(const glm::vec3& position = glm::vec3(0), const glm::vec3& velocity = glm::vec3(0)) noexcept 
		: transform(position, glm::quat(glm::vec3(0.25f))), velocity(velocity), fireDelay(0), fireCountdown(0) {}

	// TODO: Encapsulate drawing functions

	void Update(Input::Keyboard input) noexcept;

	inline Model GetModel() const noexcept
	{
		return { this->transform };
	}

	inline glm::vec3 GetVelocity() const noexcept
	{
		return this->velocity;
	}

	void Draw(Shader& shader, VAO& vertex, MeshData& renderData, Model model) const noexcept;
};


#endif // PLAYER_CONTROLLER_H
