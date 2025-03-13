#include "Satelite.h"
#include "Model.h"
#include "VertexArray.h"
#include "Vertex.h"

// Order is messed up so currently is laid out like
// 1. Body
// 2. Solar Panel +X
// 3. Antenna
// 4. Solar Panel -X
MeshData datum;

VAO VertexFormat;

// These were trial and error'd
static constexpr float SateliteSize = 0.5f;
static constexpr float CapsuleInner = 3.25f;
static constexpr float CapsuleRadius = 0.75f;

void Satelite::Draw(Shader& shader) const noexcept
{
    if (!VertexFormat.GetArray())
    {
        VertexFormat.ArrayFormatOverride<glm::vec3>(0, 0, 0, 0);
        VertexFormat.ArrayFormatOverride<glm::vec3>(1, 0, 0, offsetof(MeshVertex, normal));
        VertexFormat.ArrayFormatOverride<glm::vec2>(2, 0, 0, offsetof(MeshVertex, texture));
    }
    Model drawModel(this->transform, SateliteSize);
    shader.SetActiveShader();
    shader.SetMat4("modelMat", drawModel.GetModelMatrix());
    shader.SetMat4("normalMat", drawModel.GetNormalMatrix());
    VertexFormat.BindArrayBuffer(datum.vertex);
    datum.index.BindBuffer();
    shader.MultiDrawElements(datum.indirect, 2);

    drawModel.rotation = drawModel.rotation * glm::angleAxis(this->solarAngle, glm::vec3(1.f, 0.f, 0.f));
    shader.SetMat4("modelMat", drawModel.GetModelMatrix());
    shader.SetMat4("normalMat", drawModel.GetNormalMatrix());
    shader.MultiDrawElements(datum.indirect, 2, 2);
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
    datum = OBJReader::ReadOBJSimple("Models\\Satelite2.obj");
    return datum.indirect.GetElementCount() == 4;
}
