#include "ShieldGenerator.h"
#include "../ResourceBank.h"
#include <ranges>


void ShieldGenerator::Draw() const noexcept
{
	Shader& shader = ShaderBank::Get("defer");
	shader.SetActiveShader();
	shader.SetVec3("shapeColor", glm::vec3(0.8f));
	VAO& truth = VAOBank::Get("new_mesh");

	ShieldGenerator::models.Bind(truth);
	truth.BindArrayBuffer(Bank<ArrayBuffer>::Get("Shields"), 1);
	shader.MultiDrawElements(ShieldGenerator::models.indirect);
}

void ShieldGenerator::Update() noexcept
{
	// Check which, if any, of the objects this is currently shielding have left the area, if they have, boot 'em out
}

std::vector<glm::vec3> ShieldGenerator::GetPoints(std::vector<glm::vec3> ins) noexcept
{
	std::vector<glm::vec3> outs{};
	// Pretend this->transform is the operand
	std::ranges::copy_if(ins, std::back_inserter(outs),
		[](const auto& in)
		{
			return glm::distance(in, glm::vec3(0.f, 50.f, 0.f)) < 30.f;
		}
	);
	return outs;
}


void ShieldGenerator::Setup()
{
	ShieldGenerator::models = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Shield.glb");
	ArrayBuffer& ref = Bank<ArrayBuffer>::Get("Shields");
	std::array<MeshMatrix, 1> paired{};
	paired[0].model = glm::scale(glm::translate(glm::vec3(0.f, 50.f, 0.f)), glm::vec3{ 10.f });
	ref.BufferData(paired);
}
