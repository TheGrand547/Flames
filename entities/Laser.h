#pragma once
#ifndef LASER_H
#define LASER_H

#include <optional>
#include "../glmHelp.h"
#include "../Lines.h"
#include "../CollisionTypes.h"

namespace Laser
{
	enum class HitType
	{
		Miss, Terrain, Shield, Entity, Player
	};
	struct Result
	{
		glm::vec3 start{}, end{};
		HitType type = HitType::Miss;
		// Will not exist if type is Miss
		std::optional<RayCollision> hit = std::nullopt;
	};

	Result FireLaserPlayer(Ray ray, float maxLength);
	Result FireLaserEnemy(Ray ray, float maxLength);
}


#endif // LASER_H