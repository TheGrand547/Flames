#include "Level.h"

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

	glm::vec3 GetInterest()
	{
		return interesting;
	}
}
