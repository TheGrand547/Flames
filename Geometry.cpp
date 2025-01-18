#include "Geometry.h"

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

namespace Cube
{
    static const std::array<glm::vec3, 8> Points =
    {
        glm::vec3{-1, -1, -1},   // -x, -y, -z
        glm::vec3{ 1, -1, -1},   // +x, -y, -z
        glm::vec3{ 1,  1, -1},   // +x, +y, -z
        glm::vec3{-1,  1, -1},   // -x, +y, -z
        glm::vec3{-1, -1,  1},   // -x, -y, +z
        glm::vec3{ 1, -1,  1},   // +x, -y, +z
        glm::vec3{ 1,  1,  1},   // +x, +y, +z
        glm::vec3{-1,  1,  1},   // -x, +y, +z
    };

    std::array<unsigned char, 24> Lines =
    {
        0, 1,  1, 2,  2, 3,  3, 0,
        4, 5,  5, 6,  6, 7,  7, 4,
        2, 6,  5, 1,
        3, 7,  4, 0,
    };

    // I'm not really sure what I was on about here, but it appears to worked for my purposes
    // If j = (index) % 6, then j = 0/4 are unique, j = 1/2 are repeated as 3/5 respectively
    static const std::array<unsigned char, 36> Triangles =
    {
        0, 4, 3, // -X Face
	    4, 7, 3,
	    0, 1, 4, // -Y Face
	    1, 5, 4,
	    1, 0, 2, // -Z Face
	    0, 3, 2,
	    6, 5, 2, // +X Face
	    5, 1, 2,
	    6, 2, 7, // +Y Face
	    2, 3, 7,
	    7, 4, 6, // +Z Face
	    4, 5, 6,
    };

    // TODO: Fix
    std::array<TextureVertex, 36> UVPoints =
        [](auto verts, auto index)
        {
            std::array<TextureVertex, 36> temp{};
            for (int i = 0; i < temp.size(); i++)
            {
                temp[i].coordinates = verts[index[i]];
            }
            return temp;
        } (Points, Triangles);

    std::array<glm::vec3, 8> GetPoints() noexcept
    {
        return Points;
    }

    std::array<TextureVertex, 36> GetUVPoints() noexcept
    {
        return UVPoints;
    }

    std::array<unsigned char, 24> GetLineIndex() noexcept
    {
        return Lines;
    }

    std::array<unsigned char, 36> GetTriangleIndex() noexcept
    {
        return Triangles;
    }
}

namespace Plane
{
    static const std::array<glm::vec3, 4> Points =
    {
        glm::vec3{ 1, 0,  1},
        glm::vec3{ 1, 0, -1},
        glm::vec3{-1, 0,  1},
        glm::vec3{-1, 0, -1}
    };

    static const std::array<TextureVertex, 4> UVPoints =
    {
        TextureVertex{glm::vec3( 1, 0,  1), glm::vec2(1, 1)},
        TextureVertex{glm::vec3( 1, 0, -1), glm::vec2(1, 0)},
        TextureVertex{glm::vec3(-1, 0,  1), glm::vec2(0, 1)},
        TextureVertex{glm::vec3(-1, 0, -1), glm::vec2(0, 0)}
    };

    static const std::array<unsigned char, 5> Line =
    {
        0, 1, 3, 2, 0
    };

    static const std::array<unsigned char, 6> Triangle =
    {
        0, 1, 2, 2, 1, 3
    };

    std::array<glm::vec3, 4> GetPoints() noexcept
    {
        return Points;
    }

    std::array<unsigned char, 5> GetLineIndex() noexcept
    {
        return Line;
    }

    std::array<unsigned char, 6> GetTriangleIndex() noexcept
    {
        return Triangle;
    }

    std::array<TextureVertex, 4> GetUVPoints() noexcept
    {
        return UVPoints;
    }
}
