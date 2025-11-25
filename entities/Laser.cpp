#include "Laser.h"

#include "../Level.h"
#include "../ResourceBank.h"
#include "../ShipManager.h"

namespace Laser
{
	void ZeroDefault(Result& out, Ray ray, float maxLength)
	{
		out = Result{ ray.point, ray.point + ray.direction * maxLength, HitType::Miss , std::nullopt };
	}

	void FireLaserTerrain(Result& out, Ray ray)
	{
		RayCollision currentHit;

		for (const auto& tri : Level::GetTriangleTree().RayCast(ray))
		{
			RayCollision temp{};
			if (tri->RayCast(ray, temp) && temp.depth < currentHit.depth)
			{
				out.type = HitType::Terrain;
				currentHit = temp;
			}
		}
		if (out.type != HitType::Miss)
		{
			out.end = currentHit.point;
			out.hit = std::make_optional(currentHit);
		}
	}

	void FireLaserBlokes(Result& out, Ray ray)
	{
		Level::GetShips().LaserCast(out, ray);
		/*
		RayCollision currentHit;
		if (out.hit.has_value())
		{
			currentHit = out.hit.value();
		}
		else
		{
			currentHit.point = out.end;
			currentHit.distance = glm::distance(out.end, out.start);
		}

		Level::GetShips();

		if (out.type == HitType::Entity)
		{
			out.end = currentHit.point;
			out.hit = std::make_optional(currentHit);
		}*/
	}
	
	void FireLaserShields(Result& out, Ray ray)
	{
		RayCollision intermidate{};
		if (out.hit.has_value())
		{
			intermidate = out.hit.value();
		}
		else
		{
			intermidate.point = out.end;
			intermidate.distance = glm::distance(out.end, out.start);
		}
		const std::vector<glm::vec3>& shieldPoints = Bank<std::vector<glm::vec3>>::Get("Spheres");
		for (glm::vec3 point : shieldPoints)
		{
			RayCollision hit{};
			Sphere bogus{ point, 4.f * Bank<float>::Get("TickTockBrain")};
			if (bogus.RayCast(ray, hit) && hit.depth < intermidate.depth)
			{
				intermidate = hit;
				out.type = HitType::Shield;
			}
		}
		if (out.type == HitType::Shield)
		{
			out.end = intermidate.point;
			out.hit = std::make_optional(intermidate);
		}
	}


	Result FireLaserPlayer(Ray ray, float maxLength)
	{
		Result procedulal{};
		ZeroDefault(procedulal, ray, maxLength);
		FireLaserTerrain(procedulal, ray);
		FireLaserShields(procedulal, ray);
		FireLaserBlokes(procedulal, ray);
		return procedulal;
	} 

	Result FireLaserEnemy(Ray ray, float maxLength)
	{
		return Result();
	}
}