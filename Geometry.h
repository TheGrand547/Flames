#pragma once
#ifndef TETRA_H
#define TETRA_H
#include <array>
#include "glmHelp.h"
#include "Vertex.h"

namespace Tetrahedron
{
	std::array<glm::vec3, 4> GetPoints() noexcept;
	std::array<unsigned char, 12> GetLineIndex() noexcept;
	std::array<unsigned char, 12> GetTriangleIndex() noexcept;
}

namespace Cube
{
	std::array<glm::vec3, 8> GetPoints() noexcept;
	std::array<unsigned char, 24> GetLineIndex() noexcept;
	std::array<unsigned char, 36> GetTriangleIndex() noexcept;
	std::array<TextureVertex, 36> GetUVPoints() noexcept;
}

namespace Planes
{
	std::array<glm::vec3, 4> GetPoints() noexcept;
	std::array<unsigned char, 5> GetLineIndex() noexcept;
	// Redundant, since the plane is ordered to be drawn as a Triangle Strip, but included for completeness
	std::array<unsigned char, 6> GetTriangleIndex() noexcept;
	std::array<TextureVertex, 4> GetUVPoints() noexcept;
}

#endif // TETRA_H