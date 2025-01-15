#pragma once
#ifndef TETRA_H
#define TETRA_H
#include <array>
#include "glmHelp.h"

namespace Tetrahedron
{
	std::array<glm::vec3, 4> GetPoints() noexcept;
	std::array<unsigned char, 12> GetLineIndex() noexcept;
	std::array<unsigned char, 12> GetTriangleIndex() noexcept;
}

#endif // TETRA_H