#include "Satelite.h"
#include "Model.h"
#include "VertexArray.h"
#include "Vertex.h"

// Order is messed up so currently is laid out like
// 1. Body
// 2. Solar Panel +X
// 3. Antenna
// 4. Solar Panel -X
std::vector<MeshPair> VertexData;
VAO VertexFormat;

// These were trial and error'd
static constexpr float SateliteSize = 0.5f;
static constexpr float CapsuleInner = 3.25f;
static constexpr float CapsuleRadius = 0.75f;

void Satelite::Draw( Shader& shader) const noexcept
{
    // TODO: Unhack this
    if (!VertexFormat.GetArray())
    {
        VertexFormat.ArrayFormat<MeshVertex>(shader);
    }
    Model drawModel(this->transform, SateliteSize);
    // TODO: Batch this into a multi-draw arrays
    shader.SetActiveShader();
    shader.SetMat4("modelMat", drawModel.GetModelMatrix());
    shader.SetMat4("normalMat", drawModel.GetNormalMatrix());
    VertexFormat.BindArrayBuffer(VertexData[0].vertex);
    shader.DrawElements(VertexData[0].index); // Draw Body
    VertexFormat.BindArrayBuffer(VertexData[2].vertex);
    shader.DrawElements(VertexData[2].index); // Draw Antenna
    drawModel.rotation = drawModel.rotation * glm::angleAxis(this->solarAngle, glm::vec3(1.f, 0.f, 0.f));
    shader.SetMat4("modelMat", drawModel.GetModelMatrix());
    shader.SetMat4("normalMat", drawModel.GetNormalMatrix());
    VertexFormat.BindArrayBuffer(VertexData[1].vertex);
    shader.DrawElements(VertexData[1].index);
    VertexFormat.BindArrayBuffer(VertexData[3].vertex);
    shader.DrawElements(VertexData[3].index);
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
    OBJReader::ReadOBJ("Models\\Satelite.obj", VertexData);
    return VertexData.size() == 4;
}
