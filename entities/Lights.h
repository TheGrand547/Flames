#pragma once
#ifndef LIGHTS_H
#define LIGHTS_H
#include "../glmHelp.h"
#include <glm/gtx/vec_swizzle.hpp>

struct LightVolume
{
	// Position position.w means point light, being the radius
	// Negative position.w means cone, with the absolute value of it being the height
	// 0 position.w means directed light
	glm::vec4 position{ glm::vec3(0.f), 10.f };
	glm::vec4 color{ 1.f };
	glm::vec4 constants{ 1.f, 0.025f, 0.f, 0.f };
	glm::vec4 direction{ 0.f };
};

struct BigLightVolume
{
	glm::vec4 position{ -66.6f };
	glm::vec4 positionViewSpace{ -66.6f };
	glm::vec4 color{ -66.6f };
	glm::vec4 constants{ -66.6f };
	glm::vec4 direction{ -66.6f };
	glm::vec4 directionViewSpace{ -66.6f };
};

inline void ConeLightingInfo(LightVolume& in, float height, float fieldOfView)
{
	// This is A/H
	float cosine = glm::cos(glm::radians(fieldOfView / 2.f));
	float coneRadius = height * glm::tan(glm::radians(fieldOfView / 2.f));
	in.position.w = -height;
	in.constants.w = coneRadius;
	in.direction.w = cosine;
}

inline BigLightVolume MakeBig(const LightVolume& smallLight, glm::mat4 transformer)
{
	BigLightVolume bigLight;
	bigLight.position = smallLight.position;
	bigLight.color = smallLight.color;
	bigLight.constants = smallLight.constants;
	bigLight.direction = smallLight.direction;

	glm::vec3 tempA = transformer * glm::vec4(glm::xyz(smallLight.position), 1.f);
	glm::vec3 tempB = transformer * glm::vec4(glm::xyz(smallLight.direction), 0.f);
	bigLight.positionViewSpace = glm::vec4(tempA, smallLight.position.w);
	bigLight.directionViewSpace = glm::vec4(tempB, smallLight.direction.w);
	return bigLight;
}

#endif // LIGHTS_H