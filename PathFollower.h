#pragma once
#ifndef PATH_FOLLOWER_H
#define PATH_FOLLOWER_H
#include "BasicPhysics.h"
#include "Capsule.h"
#include "kdTree.h"
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
	PathFollower() noexcept;
	PathFollower(const glm::vec3& position, const float& mass = 1.f) noexcept;
	~PathFollower() noexcept;

	PathFollower& operator=(const PathFollower& other) noexcept;


	inline AABB GetAABB() const noexcept { return this->box.GetAABB(); }
	inline glm::vec3 GetPosition() const noexcept { return this->box.Center(); }

	inline glm::mat4 GetModelMatrix()  const noexcept { return this->box.GetNormalMatrix(); }
	inline glm::mat4 GetNormalMatrix() const noexcept { return this->box.GetNormalMatrix(); }

	void Update() noexcept;

	void PathUpdate() noexcept;
	void Collision() noexcept;

	// Bandaid till proper pathfinding/collision hierarchy is written
	static ArrayBuffer latestPathBuffer;
};

#endif // PATH_FOLLOWER_H
