#include "glmHelp.h"

glm::mat3 SetForward(const glm::vec3& vec, const glm::vec3& up) noexcept
{
    glm::mat3 mat{ 1.f };
    mat[0] = glm::normalize(vec);
    mat[2] = glm::normalize(glm::cross(up, mat[0]));
    mat[1] = glm::normalize(glm::cross(mat[0], mat[2]));
    return mat;
}
