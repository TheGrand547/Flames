#include "Cube.h"
#include <array>
#include <glew.h>
#include "glm/glm.hpp"

static GLuint cubeBuffer;

static std::array<glm::vec3, 8> plainCubeVerts{
	{
		{-1, -1, -1},
		{ 1, -1, -1},
		{ 1,  1, -1},
		{-1,  1, -1},
		{-1, -1,  1},
		{ 1, -1,  1},
		{ 1,  1,  1},
		{-1,  1,  1},
	}
};
