#pragma once
#ifndef DEBRIS_MANAGER_H
#define DEBRIS_MANAGER_H
#include <vector>
#include "Transform.h"
#include "Shader.h"

class DebrisManager
{
private:
	struct Debris
	{
		Transform transform, delta;
		unsigned char drawIndex;
		std::uint16_t ticksAlive;
		float scale;
	};

	// TODO: Maybe an array or circular buffer to cap size, who knows
	std::vector<Debris> debris;
	bool dirty = true, superDirty = true;
public:
	DebrisManager() noexcept = default;

	void Update() noexcept;
	void Draw(Shader& shader) noexcept;


	void FillBuffer() noexcept;
	void AddDebris(glm::vec3 postion, glm::vec3 velocity) noexcept;

	static bool LoadResources() noexcept;
};
#endif // DEBRIS_MANAGER_H
