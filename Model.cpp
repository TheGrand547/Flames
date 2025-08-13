#include "Model.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>


Model Model::ApplyParent(const Model& parent) noexcept
{
    Model temp{ *this };
    temp.scale *= parent.scale;
    glm::vec3 immediate = temp.rotation * temp.translation;
    temp.rotation = parent.rotation * temp.rotation;
    //temp.rotation = temp.rotation * parent.rotation;
    temp.translation = parent.translation + immediate;
    return temp;
}

Model Model::ApplyParentInGlobal(const Model& parent) noexcept
{
    Model temp{ *this };
    temp.scale *= parent.scale;
    temp.rotation = parent.rotation * temp.rotation;
    temp.translation = parent.translation + temp.translation;
    return temp;
}

glm::mat4 Model::GetModelMatrix() const noexcept
{
    return glm::scale(glm::translate(glm::mat4(1.f), this->translation) * glm::mat4_cast(this->rotation), this->scale);
}

glm::mat4 Model::GetNormalMatrix() const noexcept
{
    return glm::mat4_cast(this->rotation);
}

MeshMatrix Model::GetMatrixPair() const noexcept
{
    glm::mat4 normal = this->GetNormalMatrix();
    return {glm::scale(glm::translate(glm::mat4(1.f), this->translation) * normal, this->scale), normal};
}
