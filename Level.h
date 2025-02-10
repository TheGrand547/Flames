#pragma once
#ifndef LEVEL_H
#define LEVEL_H
#include "kdTree.h"
#include "StaticOctTree.h"
#include "DynamicTree.h"
#include "PathNode.h"
#include "Bullet.h"

namespace Level
{
	static StaticOctTree<OBB> Geometry;
	static kdTree<PathNodePtr> Tree;
	static std::vector<PathNodePtr> AllNodes;
	void Clear() noexcept;

	Bullet& AddBullet(const glm::vec3& position, const glm::vec3& velocity);
	std::vector<Bullet>& GetBullets();

	// Points of Interest
	std::vector<glm::vec3>& GetPOI();
}


#endif // LEVEL_H