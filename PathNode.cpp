#include "PathNode.h"

PathNode::PathNode(const glm::vec3& position) : position(position), nodes(), distances()
{

}

float PathNode::distance(const PathNode& other) const noexcept
{
    auto location = this->distances.find(other.shared_from_this());
    if (location != this->distances.end())
    {
        return location->second;
    }
    return std::numeric_limits<float>::infinity();
}

std::vector<std::weak_ptr<PathNode>> PathNode::neighbors() const
{
    return this->nodes;
}

bool PathNode::addNeighborUnconditional(std::shared_ptr<PathNode>& A, std::shared_ptr<PathNode>& B) noexcept
{
    bool AHasB = A->distances.contains(B);
    bool BHasA = B->distances.contains(A);
    if (AHasB && BHasA)
    {
        return false;
    }
    if (!AHasB)
    {
        A->nodes.push_back(B);
        A->distances[B] = glm::distance(A->position, B->position);
    }
    if (!BHasA)
    {
        B->nodes.push_back(A);
        B->distances[A] = glm::distance(A->position, B->position);
    }
    return true;
}

std::shared_ptr<PathNode> PathNode::MakeNode(const glm::vec3& position)
{
    return std::make_shared<PathNode>(PathNode{position});
}
