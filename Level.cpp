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

	static std::vector<Bullet> Bullets;

	Bullet& AddBullet(const glm::vec3& position, const glm::vec3& velocity)
	{
		return Bullets.emplace_back(position, velocity);
	}

	std::vector<Bullet>& GetBullets()
	{
		return Bullets;
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
