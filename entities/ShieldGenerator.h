#pragma once
#ifndef SHIELD_GENERATOR_H
#define SHIELD_GENERATOR_H
#include "../glmHelp.h"
#include "../Transform.h"
#include "../OBJReader.h"

struct ShieldGenerator
{
	static inline MeshData models;
	Transform transform;
	std::uint16_t cycle{ 0 };

	void Draw() const noexcept;
	void Update() noexcept;

	std::vector<glm::vec3> GetPoints(std::vector<glm::vec3> ins) noexcept;

	static void Setup();
};



#endif // SHIELD_GENERATOR_H