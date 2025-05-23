#include "Satelite.h"
#include "Model.h"
#include "VertexArray.h"
#include "Vertex.h"
#include "ResourceBank.h"
#include "glUtil.h"

// Order is messed up so currently is laid out like
// 1. Body
// 2. Solar Panel +X
// 3. Antenna
// 4. Solar Panel -X
static MeshData datum;

static VAO VertexFormat;

// These were trial and error'd
static constexpr float SateliteSize = 0.5f;
static constexpr float CapsuleInner = 3.25f;
static constexpr float CapsuleRadius = 0.75f;

void Satelite::Draw(Shader& shader) const noexcept
{
	if (!VertexFormat.GetArray())
	{
		VertexFormat.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0, sizeof(NormalMeshVertex));
		VertexFormat.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(NormalMeshVertex, normal),  sizeof(NormalMeshVertex));
		VertexFormat.ArrayFormatOverride<glm::vec2>(2, 0, 0, offsetof(NormalMeshVertex, texture), sizeof(NormalMeshVertex));
	}
	//DisableGLFeatures<FaceCulling>();
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	Model drawModel(this->transform, SateliteSize * 2.f);
	shader.SetActiveShader();
	datum.Bind(VertexFormat);
	VertexFormat.BindArrayBuffer(datum.vertex, 1);
	datum.index.BindBuffer();
	datum.indirect.BindBuffer();
	shader.SetVec3("shapeColor", glm::vec3(1.f, 0.f, 0.f));
	shader.SetMat4("modelMat", drawModel.GetModelMatrix());
	shader.SetMat4("normalMat", drawModel.GetNormalMatrix());
	//shader.DrawElements(datum.indirect, 1);
	shader.MultiDrawElements(datum.indirect, 2);

	drawModel.rotation = drawModel.rotation * glm::angleAxis(this->solarAngle, glm::vec3(1.f, 0.f, 0.f));
	shader.SetMat4("modelMat", drawModel.GetModelMatrix());
	shader.SetMat4("normalMat", drawModel.GetNormalMatrix());
	shader.MultiDrawElements(datum.indirect, 2, 2);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//EnableGLFeatures<FaceCulling>();
}

void Satelite::Update() noexcept
{
	this->solarAngle += Tick::TimeDelta * glm::quarter_pi<float>() * 0.5f;
}

Capsule Satelite::GetBounding() const noexcept
{
	constexpr float Delta = 0.5f * CapsuleInner;
	const glm::vec3 forward = static_cast<glm::mat3>(this->transform.rotation)[0] * Delta;

	return Capsule({ this->transform.position + forward, this->transform.position - forward }, CapsuleRadius);
}

bool Satelite::LoadResources() noexcept
{
	datum = OBJReader::MeshThingy<NormalMeshVertex>("Models\\Satelite.glb");
	return datum.indirect.GetElementCount() == 4;
}
