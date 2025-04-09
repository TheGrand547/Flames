#include "ShipManager.h"
#include "Level.h"
#include "ResourceBank.h"
#include "Parallel.h"
#include <ranges>
#include "Input.h"

void ShipManager::Update() noexcept
{
	this->inactive.clear();
	this->inactive.reserve(this->brainDrain2.size());

	// Arbitrary threshold
	if (!Input::Mouse::CheckButton(Input::Mouse::ButtonMiddle) && this->brainDrain2.size() > 50)
	{
		for (auto i = 0; i < this->brainDrain2.size(); i++)
			this->inactive.push_back({});
		std::ranges::iota_view viewing(static_cast<std::size_t>(0), static_cast<std::size_t>(this->brainDrain2.size()));
		std::for_each(std::execution::par, viewing.begin(), viewing.end(), [&](std::size_t i)
			{
				ClockBrain& element = this->brainDrain2[i];
				glm::vec3 position = element.GetPos();
				element.Update();
				this->inactive[i] = (element.GetPair());

				// Keeping this in just in case the issue returns, despite the performance penalty
				auto& p = this->inactive[i];
				if (glm::any(glm::greaterThanEqual(glm::abs(p.model[0]), glm::vec4(10.f))))
				{
					Log("Big Trouble");
				}
			}
		);
	}
	else
	{
		std::for_each(this->brainDrain2.begin(), this->brainDrain2.end(), [&](ClockBrain& element)
			{
				glm::vec3 position = element.GetPos();
				element.Update();
				this->inactive.push_back(element.GetPair());
			}
		);
	}
	std::swap(this->active, this->inactive);

	Parallel::erase_if(std::execution::par, this->brainDrain2, 
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
