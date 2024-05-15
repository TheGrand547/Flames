#include "Model.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>


glm::mat4 Model::GetModelMatrix() const noexcept
{
    //glm::quat quat(glm::radians(this->rotation));
    //return glm::scale(glm::translate(glm::mat4(1.f), this->translation) * (glm::mat4) quat, this->scale);
    glm::vec3 local = glm::radians(this->rotation);
    return glm::scale(glm::translate(glm::mat4(1.f), this->translation) * glm::eulerAngleXYZ(local.x, local.y, local.z), this->scale);
}

glm::mat4 Model::GetNormalMatrix() const noexcept
{
    glm::vec3 local = glm::radians(this->rotation);
    return glm::translate(glm::mat4(1.f), this->translation) * glm::eulerAngleXYZ(local.x, local.y, local.z);
}