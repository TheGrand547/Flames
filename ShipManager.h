#pragma once
#ifndef SHIP_MANAGER_H
#define SHIP_MANAGER_H
#include "DynamicTree.h"
#include "ClockBrain.h"
#include "Buffer.h"
#include "async/BufferSync.h"
#include "Level.h"
#include "entities/Laser.h"
#include <ranges>

class ShipManager
{
protected:
	std::vector<ClockBrain> brainDrain;
	ArrayBuffer pain;
	ArrayBuffer smooth;
	
	BufferSync<std::vector<MeshMatrix>> active;
	std::vector<MeshMatrix> inactive;
	BufferSync<std::vector<Bundle<glm::vec3>>> fools;
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
	inline ArrayBuffer& GetPositions() noexcept
	{
		return this->smooth;
	}
	inline std::vector<Bundle<glm::vec3>> GetRawPositions() noexcept
	{
		return this->fools.ExclusiveOperation([&](auto& list)
			{
				std::vector<Bundle<glm::vec3>> out;
				std::ranges::copy(list, std::back_inserter(out));
				return out;
			}
		);
	}

	inline std::vector<glm::mat4> GetOBBS() noexcept
	{
		std::vector<glm::mat4> outs;
		std::ranges::copy(this->brainDrain | 
			std::views::transform([](const auto& a) {return a.GetOBB().GetModelMatrix(); }),
			std::back_inserter(outs)
		);
		return outs;
	}

	// Calling this assumes that everything that could've been hit before has been checked
	// This is the end of the line, if something is even closer than one of these ships, 
	// well they'll both get hit
	void LaserCast(Laser::Result& out, Ray ray);
};

#endif // SHIP_MANAGER_H