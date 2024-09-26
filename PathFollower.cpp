#include "PathFollower.h"
#include "Pathfinding.h"

std::vector<PathNodePtr> PathFollower::PathNodes;
ArrayBuffer PathFollower::latestPathBuffer;
kdTree<PathNodePtr> PathFollower::pathNodeTree;

PathFollower::PathFollower() noexcept : physics(glm::vec3(0), 1.f), box(glm::vec3(0), glm::vec3(0.5, 2, 0.5))
{
	this->box.ReCenter(this->physics.position);
	this->capsule.SetCenter(this->physics.position);
}

PathFollower::PathFollower(const glm::vec3& position, const float& mass) noexcept : physics(position, mass), 
																	box(glm::vec3(0), glm::vec3(0.5, 2, 0.5))
{
	this->box.ReCenter(position);
	this->capsule.SetCenter(position);
}

PathFollower::~PathFollower() noexcept
{
	//std::cout << "Follower Down" << std::endl;
	this->path.clear();
}

PathFollower& PathFollower::operator=(const PathFollower& other) noexcept
{
	if (this != &other)
	{
		this->path.clear();
		this->box = other.box;
		this->capsule.SetCenter(other.capsule.GetCenter());
		this->physics = other.physics;
		this->path.reserve(other.path.size());
		std::copy(other.path.begin(), other.path.end(), std::back_inserter(this->path));
	}
	return *this;
}

void PathFollower::Update() noexcept
{
	if (this->path.size() == 0)
	{
		// REGENERATE PATH
		glm::vec3 center = this->capsule.GetCenter();
		PathNodePtr start = nullptr, end = nullptr;
		float minDist = INFINITY;
		for (auto& possible : PathNodes)
		{
			if (glm::distance(center, possible->GetPosition()) < minDist)
			{
				start = possible;
				minDist = glm::distance(center, possible->GetPosition());
			}
		}
		if (start)
		{
			end = PathNodes[std::rand() % PathNodes.size()];
			this->path = AStarSearch<PathNode>(start, end,
				[](const PathNode& a, const PathNode& b)
				{
					return glm::distance(a.GetPosition(), b.GetPosition());
				}
			);
			// TODO: Figure out what the hell is happening
			std::vector<glm::vec3> positions, lines;
			positions.reserve(this->path.size());
			lines.reserve(this->path.size());
			for (std::size_t i = 0; i < this->path.size(); i++)
			{
				positions.push_back(this->path[i]->GetPosition());
				lines.push_back(this->path[i]->GetPosition());
			}
			latestPathBuffer.BufferData(positions, StaticDraw);
		}
	}
	else
	{
		this->PathUpdate();
	}
	this->Collision();
}

void PathFollower::Collision() noexcept
{
	this->capsule.SetCenter(this->physics.position);

	::Collision placeholder;
	for (auto& possible : staticBoxes.Search(this->capsule.GetAABB()))
	{
		if (possible->Overlap(this->capsule, placeholder))
		{
			this->capsule.Translate(placeholder.normal * placeholder.depth);
		}
	}
	this->box.ReCenter(this->capsule.GetCenter());
}

void PathFollower::PathUpdate() noexcept
{
	if (this->path.size() == 0)
		return;

	auto& target = this->path.back();
	glm::vec3 pos = target->GetPosition();
	// TODO: Store this with a little offset to not look so rigidly uniform in movement
	if (glm::distance(this->capsule.ClosestPoint(pos), pos) < this->capsule.GetRadius() / 2.f)
	{
		Log(glm::length(this->physics.velocity));
		// Need to move to the next node
		this->path.pop_back();
	}
	else
	{
		// Accelerate towards it
		glm::vec3 targetVelocity = glm::normalize(pos - this->capsule.GetCenter());
		glm::vec3 unitVelocity = glm::normalize(this->physics.velocity);
		
		glm::vec3 direction = glm::normalize(targetVelocity - unitVelocity);
		if (glm::any(glm::isnan(direction)))
		{
			// What?
			//Log("Error Condition thingy");
			direction = targetVelocity;
		}

		this->physics.ApplyForces(direction * 20.f, Tick::TimeDelta);
	}
	this->capsule.SetCenter(this->physics.position);
	this->box.ReCenter(this->physics.position);
	// TODO: ensure descendents do the collision
}
