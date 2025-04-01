#include "ShipManager.h"
#include "Level.h"
#include "ResourceBank.h"

void ShipManager::Update() noexcept
{
	this->inactive.clear();
	this->inactive.reserve(this->brainDrain.size());
	//this->brainDrain.for_each([&] (ClockBrain& element)
	std::for_each(this->brainDrain2.begin(), this->brainDrain2.end(), [&](ClockBrain& element)
		{
			glm::vec3 position = element.GetPos();
			element.Update();
			this->inactive.push_back(element.GetPair());
			return position != element.GetPos();
		}
	);
	std::swap(this->active, this->inactive);
	// I got confused and I think made the wrong thing a dynamic oct tree

	
	//for (Bullet& bullet : Level::GetBullets())
	//for (auto& bloke : this->brainDrain2)
	std::erase_if(this->brainDrain2, 
		[](ClockBrain& bloke)
		{
			for (auto& bullet : Level::GetBulletTree().Search(bloke.GetAABB()))
			{
				//Log(glm::distance(thingy->GetPos(), bullet.position));
				if (bullet->GetAABB().Overlap(bloke.GetAABB()))
				{
					Log("Oh shit we got one");
					bullet->transform.position = glm::vec3(NAN);
					Level::SetExplosion(bloke.GetPos());
					return true;
				}
			}
			return false;
		}
	);
}

void ShipManager::Draw(MeshData& data, VAO& vao, Shader& shader) noexcept
{
	DrawIndirect flubber = data.rawIndirect[0];
	shader.SetActiveShader();
	shader.SetVec3("shapeColor", glm::vec3(0.8f));
	flubber.instanceCount = this->pain.GetElementCount();
	data.Bind(VAOBank::Get("instance"));
	VAOBank::Get("instance").BindArrayBuffer(this->pain, 1);
	shader.DrawElements(flubber);

	/*
	for (auto& bloke : this->brainDrain)
	{
		bloke.Draw(data, vao, shader);
	}*/
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
