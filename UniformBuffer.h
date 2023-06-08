#pragma once
#ifndef UNIFORM_BUFFER_H
#define UNIFORM_BUFFER_H
#include "Buffer.h"
#include "Shader.h"

class UniformBuffer : public Buffer
{
private:
	GLuint bufferBinding;
public:
	UniformBuffer();
	~UniformBuffer();

	inline void Generate(BufferAccess access, std::size_t size = 0);
	inline void BindUniform(std::size_t offset = 0, std::size_t size = 0);
	inline void SetBindingPoint(GLuint point);
};

inline void UniformBuffer::Generate(BufferAccess access, std::size_t size)
{
	Buffer::Generate(UniformBufferObject, access, (GLsizeiptr) size);
}

inline void UniformBuffer::BindUniform(std::size_t offset, std::size_t size)
{
	if (size == 0)
	{
		size = this->length;
	}
	glBindBufferRange(GL_UNIFORM_BUFFER, this->bufferBinding, this->buffer, (GLintptr) offset, (GLsizeiptr) size);
}

inline void UniformBuffer::SetBindingPoint(GLuint point)
{
	this->bufferBinding = point;
}

#endif // UNIFORM_BUFFER_H
