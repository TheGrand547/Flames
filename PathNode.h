#pragma once
#ifndef PATH_NODE_H
#define PATH_NODE_H
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <unordered_map>
#include <vector>
#include "glmHelp.h"

template<typename T, typename W> concept ConditionFunction = requires(T t, std::shared_ptr<W>& w1, std::shared_ptr<W>& w2)
{
	{ t(w1, w2) } -> std::convertible_to<bool>;

	t(w1, w2);
};

class PathNode 
{	
protected:
	glm::vec3 position;
	std::vector<std::weak_ptr<PathNode>> nodes;
	std::unordered_map<std::shared_ptr<PathNode>, float> distances;

	PathNode(const glm::vec3& position);
public:
	std::size_t hash() const noexcept 
	{ 
		return std::hash<glm::vec3>{}(this->position);
	}
	
	constexpr bool operator==(const PathNode& other) const noexcept;

	float distance(const std::shared_ptr<PathNode>& other) noexcept;
	std::vector<std::weak_ptr<PathNode>> neighbors();


	static bool addNeighborUnconditional(std::shared_ptr<PathNode>& A, std::shared_ptr<PathNode>& B) noexcept;
	template <ConditionFunction<PathNode> Conditional> static bool addNeighbor(std::shared_ptr<PathNode&> A, std::shared_ptr<PathNode>& B, Conditional condition);
	static std::shared_ptr<PathNode> MakeNode(const glm::vec3& position);
};

namespace std
{
	template<> struct hash<PathNode>
	{
		size_t operator()(const PathNode& op) const
		{
			return op.hash();
		}
	};
}

template <ConditionFunction<PathNode> Conditional> bool PathNode::addNeighbor(std::shared_ptr<PathNode&> A, std::shared_ptr<PathNode>& B, Conditional condition)
{
	if (condition(A, B))
	{
		return PathNode::addNeighborUnconditional(A, B);
	}
	return false;
}

constexpr bool PathNode::operator==(const PathNode& other) const noexcept
{
	return this->position == other.position;
}

#endif // PATH_NODE_H
