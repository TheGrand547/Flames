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
		glm::vec3 position;
		NodeType type;
		StaticVector<IndexType> connections;
	};

	NavMesh(std::string name) noexcept;
	~NavMesh() noexcept = default;

	void Generate(std::span<const glm::vec3> points, std::function<bool(const Node&, const Node&)> function) noexcept;

	bool Load(std::string filename) noexcept;
	bool Load() noexcept;
	void Export() noexcept;

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

protected:
	std::vector<Node> nodes;
	std::string name;
};

#endif // NAV_MESH_H