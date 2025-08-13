#pragma once
#ifndef SHIP_MANAGER_H
#define SHIP_MANAGER_H
#include "DynamicTree.h"
#include "ClockBrain.h"
#include "Buffer.h"
#include "async/BufferSync.h"

class ShipManager
{
protected:
	std::vector<ClockBrain> brainDrain;
	ArrayBuffer pain;
	ArrayBuffer smooth;
	
	std::vector<MeshMatrix> active, inactive;
	BufferSync<std::vector<glm::vec3>> fools;
	bool dirty = true;
public:
	ShipManager() noexcept = default;
	~ShipManager() noexcept = default;

	inline ClockBrain& Make()
	{
		return this->brainDrain.emplace_back();
	}

	void Update() noexcept;
	void Draw(MeshData& data, VAO& vao, Shader& shader) noexcept;
	void UpdateMeshes() noexcept;
	inline glm::vec3 GetPos() 
	{
		if (this->brainDrain.size() > 0)
		{
			return this->brainDrain[0].GetPos();
		}
		return glm::vec3(1.f);
	}
	inline AABB GetAABB()
	{
		if (this->brainDrain.size() > 0)
		{
			//return this->brainDrain.begin()->GetAABB();
			return this->brainDrain[0].GetAABB();
		}
		return AABB();
	}
	inline OBB GetOBB()
	{
		if (this->brainDrain.size() > 0)
		{
			//return this->brainDrain.begin()->GetAABB();
			return this->brainDrain[0].GetOBB();
		}
		return OBB();
	}
};

#endif // SHIP_MANAGER_H