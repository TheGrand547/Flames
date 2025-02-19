#pragma once
#ifndef DRAW_STRUCT_H
#define DRAW_STRUCT_H
#include <glew.h>


struct DrawIndirect
{
	GLuint vertexCount;
	GLuint instanceCount;
	GLuint firstVertexIndex = 0;
	GLint vertexOffset     = 0;
	GLuint instanceOffset   = 0;

	constexpr DrawIndirect(GLuint vertexCount, GLuint instanceCount, GLuint firstVertexIndex, GLint vertexOffset, GLuint instanceOffset) noexcept :
		vertexCount(vertexCount), instanceCount(instanceCount), firstVertexIndex(firstVertexIndex), vertexOffset(vertexOffset),
		instanceOffset(instanceOffset) {}
};

#endif // DRAW_STRUCT_H