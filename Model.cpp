#include "Model.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>


glm::mat4 Model::GetModelMatrix() const noexcept
{
    return glm::scale(glm::translate(glm::mat4(1.f), this->translation) * glm::mat4_cast(this->rotation), this->scale);
}

glm::mat4 Model::GetNormalMatrix() const noexcept
{
    return glm::translate(glm::mat4(1.f), this->translation) * glm::mat4_cast(this->rotation);
}

MeshMatrix Model::GetMatrixPair() const noexcept
{
    glm::mat4 normal = this->GetNormalMatrix();
    return {glm::scale(normal, this->scale), normal};
}
