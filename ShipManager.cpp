#include "ShipManager.h"
#include "Level.h"
#include "ResourceBank.h"
#include "Parallel.h"
#include <ranges>
#include "Input.h"
#include "StaticVector.h"

void ShipManager::Update() noexcept
{
	this->inactive.clear();
	this->inactive.reserve(this->brainDrain.size());

	std::vector<glm::vec3> pointers;
	// Arbitrary threshold
	if (!Input::Mouse::CheckButton(Input::Mouse::ButtonMiddle) && this->brainDrain.size() > 50)
	{
		// I don't know if this is even a good idea
		StaticVector<MeshMatrix> meshes(this->brainDrain.size(), MeshMatrix({ 0.f }, { 0.f }));
		
		Parallel::for_each_index(std::execution::par, this->brainDrain, [&](std::size_t i)
			{
				ClockBrain& element = this->brainDrain[i];
				glm::vec3 position = element.GetPos();
				element.Update();
				//this->inactive[i] = (element.GetPair());
				meshes[i] = (element.GetPair());

				// Keeping this in just in case the issue returns, despite the performance penalty
				auto& p = meshes[i];
				if (glm::any(glm::greaterThanEqual(glm::abs(p.model[0]), glm::vec4(10.f))))
				{
					Log("Big Trouble");
				}
			}
		);
		std::copy(meshes.begin(), meshes.end(), std::back_inserter(this->inactive));
	}
	else
	{
		std::for_each(this->brainDrain.begin(), this->brainDrain.end(), [&](ClockBrain& element)
			{
				glm::vec3 position = element.GetPos();
				element.Update();
				this->inactive.push_back(element.GetPair());
				pointers.push_back(element.GetPos());
				pointers.push_back(element.GetPos() + glm::mat3_cast(element.GetTransform().rotation)[0] * 10.f);

			}
		);
	}
	std::swap(this->active, this->inactive);
	this->fools.Swap(pointers);
	Parallel::erase_if(std::execution::par, this->brainDrain, 
		[](ClockBrain& bloke)
		{
			for (auto& bullet : Level::GetBulletTree().Search(bloke.GetAABB()))
			{
				//Log(glm::distance(thingy->GetPos(), bullet.position));
				if (bullet->team == 0 && bullet->GetAABB().Overlap(bloke.GetAABB()))
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

	ShaderBank::Get("uniform").SetActiveShader();
	VAOBank::Get("uniform").Bind();
	VAOBank::Get("uniform").BindArrayBuffer(this->smooth);
	ShaderBank::Get("uniform").SetMat4("Model", glm::mat4(1.f));
	ShaderBank::Get("uniform").DrawArray<DrawType::Lines>(this->smooth);
}

void ShipManager::UpdateMeshes() noexcept
{
	this->pain.BufferData(this->active);
	this->fools.ExclusiveOperation([&](std::vector<glm::vec3>& p) {this->smooth.BufferData(p); });
}
