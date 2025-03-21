#include "ShipManager.h"
#include "Level.h"

void ShipManager::Update() noexcept
{
	this->inactive.clear();
	this->inactive.reserve(this->brainDrain.size());
	this->brainDrain.for_each([&] (ClockBrain& element)
		{
			glm::vec3 position = element.GetPos();
			element.Update();
			//this->inactive.push_back(element.GetPair());
			return position != element.GetPos();
		}
	);
	std::swap(this->active, this->inactive);
	// I got confused and I think made the wrong thing a dynamic oct tree

	for (Bullet& bullet : Level::GetBullets())
	{
		for (auto& thingy : this->brainDrain.Search(AABB::MakeAABB(bullet.position - bullet.velocity * Tick::TimeDelta, 
			bullet.position, 
			bullet.position + bullet.velocity * Tick::TimeDelta)))
		{
			//Log(glm::distance(thingy->GetPos(), bullet.position));
			if (glm::distance(thingy->GetPos(), bullet.position) < 0.5f)
			{
				Log("Oh shit we got one");
				bullet.position = glm::vec3(NAN);
				Level::SetExplosion(thingy->GetPos());
			}
		}
	}
}

void ShipManager::Draw(MeshData& data, VAO& vao, Shader& shader) noexcept
{
	for (auto& bloke : this->brainDrain)
	{
		bloke.Draw(data, vao, shader);
	}
	/*
	this->brainDrain.for_each([&](ClockBrain& guy) 
		{
			guy.Draw(std::forward(data), std::forward(vao), std::forward(shader)); 
			return false; 
		}
	);
	*/
}

void ShipManager::UpdateMeshes() noexcept
{
	this->pain.BufferData(this->active);
}
