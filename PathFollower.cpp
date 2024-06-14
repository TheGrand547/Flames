#include "PathFollower.h"
#include "Pathfinding.h"

PathFollower::PathFollower(const glm::vec3& position, const float& mass) noexcept : physics(position, mass)
{
	this->box.ReCenter(position);
	this->capsule.SetCenter(position);
}

PathFollower::~PathFollower() noexcept
{
	this->path.clear();
}

void PathFollower::Update(const float& timestep, std::vector<PathNodePtr>& pathNodes, StaticOctTree<OBB>& boxes, ArrayBuffer& guyNodes) noexcept
{
	if (this->path.size() == 0)
	{
		// REGENERATE PATH
		glm::vec3 center = this->capsule.GetCenter();
		PathNodePtr start = nullptr, end = nullptr;
		float minDist = INFINITY;
		for (auto& possible : pathNodes)
		{
			if (glm::distance(center, possible->GetPosition()) < minDist)
			{
				start = possible;
				minDist = glm::distance(center, possible->GetPosition());
			}
		}
		if (start)
		{
			end = pathNodes[std::rand() % pathNodes.size()];
			this->path = AStarSearch<PathNode>(start, end,
				[](const PathNode& a, const PathNode& b)
				{
					return glm::distance(a.GetPosition(), b.GetPosition());
				}
			);
			std::vector<glm::vec3> positions, lines;
			positions.reserve(this->path.size());
			lines.reserve(this->path.size());
			for (std::size_t i = 0; i < this->path.size(); i++)
			{
				positions.push_back(this->path[i]->GetPosition());
				lines.push_back(this->path[i]->GetPosition());
			}
			guyNodes.BufferData(positions, StaticDraw);
		}
	}
	else
	{
		auto& target = this->path.back();
		glm::vec3 pos = target->GetPosition();
		if (glm::distance(this->capsule.ClosestPoint(pos), pos) < this->capsule.GetRadius() / 2.f)
		{
			// Need to move to the next node
			this->path.pop_back();
			//pathTestGuy.velocity *= 0.5f;
		}
		else
		{
			// Accelerate towards it
			glm::vec3 delta = glm::normalize(pos - this->capsule.GetCenter());
			glm::vec3 unitVelocity = glm::normalize(this->physics.velocity);
			float origin = glm::dot(unitVelocity, delta);
			if (glm::abs(glm::dot(glm::normalize(this->physics.velocity), delta)) < 0.25f)
			{
				//pathTestGuy.velocity *= 0.85f;
			}
			glm::vec3 direction = glm::normalize(delta - unitVelocity);
			if (glm::any(glm::isnan(direction)))
				direction = delta;

			//if (glm::length(pathTestGuy.velocity) < 0.5f)
			this->physics.ApplyForces(direction, timestep);
			//if (glm::length(pathTestGuy.velocity) > 1.f)
			{
				//pathTestGuy.velocity = glm::normalize(pathTestGuy.velocity) * 1.f;
			}
			//pathTestGuy.velocity = delta * timeDelta;
		}
	}
	this->capsule.SetCenter(this->physics.position);

	Collision placeholder;
	for (auto& possible : boxes.Search(this->capsule.GetAABB()))
	{
		if (possible->Overlap(this->capsule, placeholder))
		{
			this->capsule.Translate(placeholder.normal * placeholder.depth);
		}
	}
	this->box.ReCenter(this->capsule.GetCenter());
}
