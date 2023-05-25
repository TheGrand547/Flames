#pragma once
#ifndef VERTEX_H
#define VERTEX_H
#include <glm/glm.hpp>

struct ColoredVertex
{
	glm::vec3 position, color;
};

struct TextureVertex
{
	glm::vec3 position;
	glm::vec2 coordinates;
};

struct MeshVertex
{
	glm::vec3 position, normal;
	glm::vec2 texture;
};

struct CompleteVertex
{
	glm::vec3 position, color, normal;
	glm::vec2 texture;
};

#endif // VERTEX_H