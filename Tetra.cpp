#include "Tetra.h"

namespace Tetrahedron
{
    static const std::array<glm::vec3, 4> Points =
    {
        glm::vec3( 0.5,    0, -glm::sqrt(2.f) / 4.f),
        glm::vec3(-0.5,    0, -glm::sqrt(2.f) / 4.f),
        glm::vec3(   0,  0.5,  glm::sqrt(2.f) / 4.f),
        glm::vec3(   0, -0.5,  glm::sqrt(2.f) / 4.f),
    };



    static const std::array<unsigned char, 12> Lines =
    {
        0, 1,
        0, 2,
        0, 3,
        1, 2,
        1, 3,
        2, 3
    };

    static const std::array<unsigned char, 12> Triangles =
    {
        0, 1, 2,
        0, 2, 3,
        0, 3, 1,
        1, 3, 2

    };

    std::array<glm::vec3, 4> GetPoints() noexcept
    {
        return Points;
    }

    std::array<unsigned char, 12> GetLineIndex() noexcept
    {
        return Lines;
    }

    std::array<unsigned char, 12> GetTriangleIndex() noexcept
    {
        return Triangles;
    }
}
