#pragma once
#ifndef LEVEL_H
#define LEVEL_H
#include "kdTree.h"
#include "StaticOctTree.h"
#include "DynamicTree.h"
#include "PathNode.h"
#include "Bullet.h"
#include "NavMesh.h"
#include "entities/EntityBundle.h"

class ShipManager;

namespace Level
{
	static StaticOctTree<OBB> Geometry;
	static kdTree<PathNodePtr> Tree;
	//static std::vector<PathNodePtr> AllNodes;
	void Clear() noexcept;

	void AddOBB(OBB obb);
	StaticOctTree<OBB>& GetOBBTree();

	using GeometryType = StaticOctTree<Triangle>;
	void AddTri(Triangle triangle);
	GeometryType& GetTriangleTree();

	Bullet& AddBullet(const glm::vec3& position, const glm::vec3& velocity);
	Bullet& AddBulletTree(const glm::vec3& position, const glm::vec3& velocity, glm::vec3 up, unsigned int team = 0);
	std::vector<Bullet>& GetBullets();

	DynamicOctTree<Bullet>& GetBulletTree();
	std::vector<PathNodePtr>& AllNodes();

	ShipManager& GetShips();

	// Points of Interest
	std::vector<glm::vec3>& GetPOI();
	glm::vec3 GetInterest();
	void SetInterest(glm::vec3 vec);

	void SetExplosion(glm::vec3 location);
	std::size_t NumExplosion();
	std::vector<glm::vec3> GetExplosion();

	NavMesh& GetNavMesh() noexcept;

	std::size_t GetCurrentTick() noexcept;
	void ResetCurrentTick() noexcept;
	void IncrementCurrentTicK() noexcept;

	// TODO: Stop doing this horseshit
	glm::vec3 GetPlayerPos() noexcept;
	void SetPlayerPos(glm::vec3 vec) noexcept;

	glm::vec3 GetPlayerVel() noexcept;
	void SetPlayerVel(glm::vec3 vec) noexcept;


	IDType GetID() noexcept;

	static inline std::unordered_map<IDType, std::int32_t> ShieldMapping;
}


#endif // LEVEL_H