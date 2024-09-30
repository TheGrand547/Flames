#pragma once
#ifndef PATH_NODE_H
#define PATH_NODE_H
#include <unordered_map>
#include <vector>
#include "glmHelp.h"

template<typename T, typename W> concept ConditionFunction = requires(T t, std::shared_ptr<W>& w1, std::shared_ptr<W>& w2)
{
	{ t(w1, w2) } -> std::convertible_to<bool>;

	t(w1, w2);
};

class PathNode : public std::enable_shared_from_this<PathNode>
{	
protected:
	glm::vec3 position;
	std::vector<std::weak_ptr<PathNode>> nodes;
	std::unordered_map<std::shared_ptr<const PathNode>, float> distances;

	PathNode(const glm::vec3& position);
public:
	std::size_t hash() const noexcept;
	
	inline glm::vec3 GetPosition() const noexcept;
	inline glm::vec3 GetPos() const noexcept;

	inline bool operator==(const PathNode& other) const noexcept;

	float distance(const PathNode& other) const noexcept;
	std::vector<std::weak_ptr<PathNode>> neighbors() const noexcept;

	bool contains(const std::shared_ptr<PathNode>& node) const noexcept;

	static bool addNeighborUnconditional(std::shared_ptr<PathNode>& A, std::shared_ptr<PathNode>& B) noexcept;
	template <ConditionFunction<PathNode> Conditional> static bool addNeighbor(std::shared_ptr<PathNode>& A, std::shared_ptr<PathNode>& B, Conditional condition);
	static std::shared_ptr<PathNode> MakeNode(const glm::vec3& position);
};

using PathNodePtr = std::shared_ptr<PathNode>;

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

template <ConditionFunction<PathNode> Conditional> bool PathNode::addNeighbor(std::shared_ptr<PathNode>& A, std::shared_ptr<PathNode>& B, Conditional condition)
{
	if (condition(A, B))
	{
		return PathNode::addNeighborUnconditional(A, B);
	}
	return false;
}

// TODO: Fix this being stupid and bad
inline glm::vec3 PathNode::GetPosition() const noexcept
{
	return this->position;
}

inline glm::vec3 PathNode::GetPos() const noexcept
{
	return this->position;
}

inline bool PathNode::operator==(const PathNode& other) const noexcept
{
	return this->position == other.position;
}

#endif // PATH_NODE_H
