#pragma once
#ifndef DEBRIS_MANAGER_H
#define DEBRIS_MANAGER_H
#include <vector>
#include "Transform.h"
#include "Shader.h"
#include "Buffer.h"

class DebrisManager
{
public:
	struct Debris
	{
		Transform transform, delta;
		unsigned char drawIndex;
		std::uint16_t ticksAlive;
		float scale;
	};
private:

	// TODO: Maybe an array or circular buffer to cap size, who knows
	std::vector<Debris> debris;
	ArrayBuffer instanceBuffer;
	DrawIndirectBuffer indirectBuffer;
	bool dirty = true, superDirty = true;
public:
	DebrisManager() noexcept = default;

	void Update() noexcept;
	void Draw(Shader& shader) noexcept;

	// Similar to ExtractElements, but just runs an update, culling any that return true from the invoking function
	// Returns the number of elements removed
	template<typename T> std::size_t Update(T function) noexcept
	{
		std::size_t removed = std::erase_if(this->debris, function);
		this->superDirty |= removed != 0;
		this->dirty = true;
		return removed;
	}


	void FillBuffer() noexcept;
	void AddDebris(glm::vec3 postion, glm::vec3 velocity) noexcept;
	

	// Applying func to each element, returns all elements that return false, removing them from the main list
	// Always invaldiates the index buffer
	template<typename T> std::vector<Debris> ExtractElements(T function) noexcept
	{
		std::vector<Debris> removed;
		auto iterator = std::remove_copy_if(this->debris.begin(), this->debris.end(), std::back_inserter(removed), function);
		this->dirty = true;
		if (removed.size() != 0)
		{
			this->debris.erase(this->debris.end() - removed.size(), this->debris.end());
			this->superDirty = true;
		}
		return removed;
	}

	void Add(std::vector<Debris>&& local) noexcept;
	void Add(std::vector<Debris>& local) noexcept;

	static bool LoadResources() noexcept;
};
#endif // DEBRIS_MANAGER_H
