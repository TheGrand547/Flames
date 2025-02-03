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
}
