#include "util.h"
#include <glm/geometric.hpp>
#include "Plane.h"

glm::vec2 GetProjectionHalfs(glm::mat4& mat)
{
	glm::vec2 result{};
	Plane rightPlane(mat[0][3] - mat[0][0], mat[1][3] - mat[1][0], mat[2][3] - mat[2][0], -mat[3][3] + mat[3][0]);
	Plane   topPlane(mat[0][3] - mat[0][1], mat[1][3] - mat[1][1], mat[2][3] - mat[2][1], -mat[3][3] + mat[3][1]);
	Plane  nearPlane(mat[0][3] + mat[0][2], mat[1][3] + mat[1][2], mat[2][3] + mat[2][2], -mat[3][3] - mat[3][2]);
	glm::vec3 local{};
	nearPlane.TripleIntersect(rightPlane, topPlane, local);
	result.x = local.x;
	result.y = local.y;
	return result;
}

// The boost implementation
void CombineHash(std::size_t& seed, const std::size_t& input)
{
	seed ^= input + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
