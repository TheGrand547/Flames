#pragma once
#ifndef DECAY_LIGHT_H
#define DECAY_LIGHT_H
#include "../glmHelp.h"
#include "Lights.h"

struct DecayLight
{
	glm::vec3 position;
	std::uint16_t lifetime, timeLeft;

	DecayLight(const glm::vec3& position, std::uint16_t lifetime = 128);

	// Decreases timeLeft by 1, returning a lightvolume. Should be removed from the list if timeleft is 0
	LightVolume Tick() noexcept;
};


#endif // DECAY_LIGHT_H