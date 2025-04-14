#pragma once
#ifndef NAV_MESH_H
#define NAV_MESH_H
#include <functional>
#include <string>
#include <vector>
#include "glmHelp.h"
#include "StaticVector.h"
#include <span>

class NavMesh
{
public:
	using IndexType = unsigned int;
	using NodeType = std::uint16_t;
	// TODO: flag bit things for NodeType
	struct Node
	{
		glm::vec3 position{ 0.f };
		NodeType type = 0;
		StaticVector<IndexType> connections;

		inline float distance(const Node& other) const noexcept
		{
			return glm::distance(this->position, other.position);
		}
	};

	NavMesh(std::string name) noexcept;
	~NavMesh() noexcept = default;

	void Generate(std::span<const glm::vec3> points, std::function<bool(const Node&, const Node&)> function) noexcept;

	void Clear() noexcept;
	bool Load(std::string filename) noexcept;
	bool Load() noexcept;
	void Export() noexcept;

	inline auto size() const noexcept
	{
		return this->nodes.size();
	}

	inline auto begin() noexcept
	{
		return this->nodes.begin();
	}

	inline auto cbegin() const noexcept
	{
		return this->nodes.cbegin();
	}

	inline auto end() noexcept
	{
		return this->nodes.end();
	}

	inline auto cend() const noexcept
	{
		return this->nodes.cend();
	}

	[[nodiscard]] std::vector<glm::vec3> AStar(IndexType start, IndexType end, 
		std::function<float(const Node&, const Node&)> heuristic) const noexcept;

protected:
	std::vector<Node> nodes;
	std::string name;
};

#endif // NAV_MESH_H