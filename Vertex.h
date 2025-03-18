#pragma once
#ifndef VERTEX_H
#define VERTEX_H
#include <glm/glm.hpp>

typedef glm::vec3 Vertex;

struct ColoredVertex
{
	glm::vec3 position, color;
};

struct UIVertex
{
	glm::vec2 position, uv;
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
		glm::vec2 texture, textureCoordinates, coordinates, uvs;
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

struct LargestVertex
{
	glm::vec3 position, color, normal, tangent;
	glm::vec2 texture;
};

struct OverstuffedVertex
{
	glm::vec3 position, color, normal, tangent, biTangent;
	glm::vec2 texture;
};

struct TangentVertex
{
	glm::vec3 tangent, biTangent;
};

#endif // VERTEX_H