#pragma once
#ifndef VERTEX_H
#define VERTEX_H
#include <glm/glm.hpp>

typedef glm::vec3 Vertex;

struct ColoredVertex
{
	glm::vec3 position, color;
};

struct NormalVertex
{
	glm::vec3 position, normal;
};

struct TextureVertex
{
	glm::vec3 position;
	union
	{
		glm::vec2 textureCoordinates, coordinates, uvs;
	};
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