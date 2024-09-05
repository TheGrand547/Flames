#include "glmHelp.h"
#include <glm/gtc/random.hpp>
#include <glm/gtx/orthonormalize.hpp>

glm::mat3 SetForward(const glm::vec3& vec, const glm::vec3& up) noexcept
{
    glm::mat3 mat{ 1.f };
    mat[0] = glm::normalize(vec);
    mat[2] = glm::normalize(glm::cross(up, mat[0]));
    mat[1] = glm::normalize(glm::cross(mat[0], mat[2]));
    return mat;
}

glm::vec3 circleRand(const float& radius) noexcept
{
    glm::vec2 values = glm::circularRand(radius);
    return glm::vec3(values.x, 0, values.y);
}

glm::vec3 circleRand(const glm::vec3& up, const float& radius) noexcept
{
    glm::vec2 values = glm::circularRand(radius);
    // TODO: don't use spherical rand, find a non-zero cross product
    glm::vec3 crossA = glm::orthonormalize(glm::sphericalRand(1.f), up);
    glm::vec3 crossB = glm::normalize(glm::cross(crossA, up));
    return values.x * crossA + values.y * crossB;
}
