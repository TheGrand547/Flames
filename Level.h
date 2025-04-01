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

	void AddOBB(OBB obb);
	StaticOctTree<OBB>& GetOBBTree();

	void AddTri(Triangle triangle);
	StaticOctTree<Triangle>& GetTriangleTree();

	Bullet& AddBullet(const glm::vec3& position, const glm::vec3& velocity);
	Bullet& AddBulletTree(const glm::vec3& position, const glm::vec3& velocity, glm::vec3 up);
	std::vector<Bullet>& GetBullets();
	DynamicOctTree<Bullet>& GetBulletTree();

	// Points of Interest
	std::vector<glm::vec3>& GetPOI();
	glm::vec3 GetInterest();
	void SetInterest(glm::vec3 vec);

	void SetExplosion(glm::vec3 location);
	std::size_t NumExplosion();
	std::vector<glm::vec3> GetExplosion();
}


#endif // LEVEL_H