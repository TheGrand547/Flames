#include "Level.h"
#include <mutex>

namespace Level
{
	void Clear() noexcept
	{
		Geometry.Clear();
		AllNodes.clear();
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

	static GeometryType Tris;

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

	Bullet& AddBulletTree(const glm::vec3& position, const glm::vec3& velocity, glm::vec3 up)
	{
		Bullet value(position, velocity, up);
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

	std::size_t NumExplosion()
	{
		return explosions.size();
	}

	glm::vec3 GetInterest()
	{
		return interesting;
	}
}
