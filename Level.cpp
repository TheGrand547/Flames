#include "Level.h"
#include <mutex>

namespace Level
{
	std::vector<PathNodePtr> AllTheNodes;
	std::vector<PathNodePtr>& AllNodes()
	{
		return AllTheNodes;
	}

	void Clear() noexcept
	{
		Geometry.Clear();
		AllNodes().clear();
		Tree.Clear();
	}

	static StaticOctTree<OBB> Boxes;
	void AddOBB(OBB obb)
	{
		Boxes.Insert(obb, obb.GetAABB());
	}
	StaticOctTree<OBB>& GetOBBTree()
	{
		return Boxes;
	}

	static GeometryType Tris(AABB(glm::vec3(1000.f)));

	void AddTri(Triangle triangle)
	{
		Tris.Insert(triangle, triangle.GetAABB());
	}

	GeometryType& GetTriangleTree()
	{
		return Tris;
	}


	static std::vector<Bullet> Bullets;

	Bullet& AddBullet(const glm::vec3& position, const glm::vec3& velocity)
	{
		return Bullets.emplace_back(position, velocity, World::Up);
	}

	std::vector<Bullet>& GetBullets()
	{
		return Bullets;
	}

	static DynamicOctTree<Bullet> bulletTree{ glm::vec3(1000.f) };

	DynamicOctTree<Bullet>& GetBulletTree()
	{
		return bulletTree;
	}

	Bullet& AddBulletTree(const glm::vec3& position, const glm::vec3& velocity, glm::vec3 up, unsigned int team)
	{
		Bullet value(position, velocity, up, team);
		return *bulletTree.Insert(value, value.GetAABB());
	}

	static std::vector<glm::vec3> PointsOfInterest;

	std::vector<glm::vec3>& GetPOI()
	{
		return PointsOfInterest;
	}
	static glm::vec3 interesting;

	void SetInterest(glm::vec3 vec)
	{
		interesting = vec;
	}


	static std::vector<glm::vec3> explosions;
	static std::mutex explosionMutex;

	void SetExplosion(glm::vec3 location)
	{
		// Maybe spawn a thread to do this on its own?
		std::lock_guard lock(explosionMutex);
		explosions.push_back(location);
	}

	std::vector<glm::vec3> GetExplosion()
	{
		std::lock_guard lock(explosionMutex);
		std::vector<glm::vec3> floops;
		std::swap(floops, explosions);
		return floops;
	}

	static NavMesh NavigationMesh("foolish");

	NavMesh& GetNavMesh() noexcept
	{
		return NavigationMesh;
	}

	static std::size_t CurrentTick = 0;

	std::size_t GetCurrentTick() noexcept
	{
		return CurrentTick;
	}

	void ResetCurrentTick() noexcept
	{
		CurrentTick = 0;
	}

	void IncrementCurrentTicK() noexcept
	{
		CurrentTick++;
	}

	std::size_t NumExplosion()
	{
		return explosions.size();
	}

	glm::vec3 GetInterest()
	{
		return interesting;
	}

	static glm::vec3 PlayerPos;

	glm::vec3 GetPlayerPos() noexcept
	{
		return PlayerPos;
	}
	
	void SetPlayerPos(glm::vec3 pos) noexcept
	{
		PlayerPos = pos;
	}


	static glm::vec3 PlayerVel;

	glm::vec3 GetPlayerVel() noexcept
	{
		return PlayerVel;
	}

	void SetPlayerVel(glm::vec3 pos) noexcept
	{
		PlayerVel = pos;
	}

	static IDType currentId;
	IDType GetID() noexcept
	{
		return currentId++;
	}
}
