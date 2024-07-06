#pragma once
#ifndef PATH_FOLLOWER_H
#define PATH_FOLLOWER_H
#include "BasicPhysics.h"
#include "Capsule.h"
#include "OrientedBoundingBox.h"
#include "PathNode.h"
#include "StaticOctTree.h"

class PathFollower
{
protected:
	BasicPhysics physics;
	Capsule capsule;
	OBB box;
	std::vector<PathNodePtr> path;
public:
	PathFollower() noexcept = default;
	PathFollower(const glm::vec3& position, const float& mass = 1.f) noexcept;
	~PathFollower() noexcept;

	PathFollower& operator=(const PathFollower& other) noexcept;

	inline glm::vec3 GetPosition() const noexcept { return this->box.Center(); }

	inline glm::mat4 GetModelMatrix()  const noexcept { return this->box.GetNormalMatrix(); }
	inline glm::mat4 GetNormalMatrix() const noexcept { return this->box.GetNormalMatrix(); }

	// TODO: Make it not this dumbass
	void Update(const float& timestep, StaticOctTree<OBB>& boxes) noexcept;

	// Bandaid till proper pathfinding/collision hierarchy is written
	static std::vector<PathNodePtr> PathNodes;
	static ArrayBuffer latestPathBuffer;
};

#endif // PATH_FOLLOWER_H
