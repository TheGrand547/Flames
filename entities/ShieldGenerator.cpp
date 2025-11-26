#include "ShieldGenerator.h"
#include "../ResourceBank.h"
#include <ranges>


void ShieldGenerator::Draw(Shader& shader, VAO& vao) const noexcept
{
	shader.SetVec3("shapeColor", glm::vec3(0.8f));

	ShieldGenerator::models.Bind(vao);
	vao.BindArrayBuffer(Bank<ArrayBuffer>::Get("Shields"), 1);
	shader.MultiDrawElements(ShieldGenerator::models.indirect);
}

void ShieldGenerator::Update() noexcept
{
	// Check which, if any, of the objects this is currently shielding have left the area, if they have, boot 'em out
}

static decltype(Level::ShieldMapping) mapping;

std::vector<glm::vec3> ShieldGenerator::GetPoints(std::vector<Bundle<glm::vec3>> ins) noexcept
{
	// Pretend this->transform is the operand
	std::vector<glm::vec3> outs{};
	std::ranges::copy(ins 
		| std::views::filter(
			[&](const auto& in)
			{
				std::int32_t value = mapping[in.id] + ((glm::distance(in.data, glm::vec3(0.f, 50.f, 0.f)) < 30.f) ? 1 : -1);
				value = std::clamp(value, static_cast<decltype(value)>(0), static_cast<decltype(value)>(Tick::PerSecond * 5));
				mapping[in.id] = value;
				return value > 50;
			}
		)
		| BundleData,
		std::back_inserter(outs));
	return outs;
}


void ShieldGenerator::Setup()
{
	ShieldGenerator::models = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Shield.glb");
	Bank<float>::Get("ShieldSize") = 10.f;

	ArrayBuffer& ref = Bank<ArrayBuffer>::Get("Shields");
	std::array<MeshMatrix, 1> paired{};
	paired[0].model = glm::scale(glm::translate(glm::vec3(-50.f, 50.f, 0.f)), glm::vec3{ 10.f });
	ref.BufferData(paired);
}
