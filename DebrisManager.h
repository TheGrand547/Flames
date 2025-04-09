#pragma once
#ifndef DEBRIS_MANAGER_H
#define DEBRIS_MANAGER_H
#include <vector>
#include "Transform.h"
#include "Shader.h"
#include "Buffer.h"
#include "Model.h"
#include "async/BufferSync.h"

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
	//std::vector<Debris> debris;
	std::vector<std::vector<Debris>> debris; // I know what I'm doing
	ArrayBuffer instanceBuffer;
	DrawIndirectBuffer indirectBuffer;
	BufferSync<std::vector<MeshMatrix>> buffered;
	bool dirty = true, superDirty = true;
	std::size_t elementCount = 0;
public:
	DebrisManager() noexcept = default;

	void Update() noexcept;
	void Draw(Shader& shader) noexcept;

	void Init() noexcept;
	inline std::size_t GetSize() const noexcept
	{
		return this->elementCount;
	}

	// Similar to ExtractElements, but just runs an update, culling any that return true from the invoking function
	// Returns the number of elements removed
	template<typename T> std::size_t Update(T function) noexcept
	{
		Log("Deprecated");
		return 0;
	}


	void FillBuffer() noexcept;
	void AddDebris(glm::vec3 postion, glm::vec3 velocity) noexcept;
	

	// Applying func to each element, returns all elements that return false, removing them from the main list
	// Always invaldiates the index buffer
	template<typename T> std::vector<Debris> ExtractElements(T function) noexcept
	{
		Log("Deprecated");
		return 0;
	}

	void Add(std::vector<Debris>&& local) noexcept;
	void Add(std::vector<Debris>& local) noexcept;

	static bool LoadResources() noexcept;
};
#endif // DEBRIS_MANAGER_H
