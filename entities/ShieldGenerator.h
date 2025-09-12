#pragma once
#ifndef SHIELD_GENERATOR_H
#define SHIELD_GENERATOR_H
#include "../glmHelp.h"
#include "../Transform.h"
#include "../OBJReader.h"
#include "EntityBundle.h"
#include <unordered_map>

struct ShieldGenerator
{
	// Maybe struct 
	struct Individual
	{
		Transform transform;
		std::uint16_t health = 1000;
	};
	static inline MeshData models;
	Transform transform;
	std::unordered_map<IDType, std::int16_t> tracking;

	void Draw() const noexcept;
	void Update() noexcept;

	std::vector<glm::vec3> GetPoints(std::vector<Bundle<glm::vec3>> ins) noexcept;

	static void Setup();
};



#endif // SHIELD_GENERATOR_H