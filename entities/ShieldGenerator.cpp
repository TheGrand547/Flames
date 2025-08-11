#include "ShieldGenerator.h"
#include "../ResourceBank.h"


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

}


void ShieldGenerator::Setup()
{
	ShieldGenerator::models = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Shield.glb");
	ArrayBuffer& ref = Bank<ArrayBuffer>::Get("Shields");
	std::array<MeshMatrix, 2> paired{};
	ref.BufferData(paired);
}
