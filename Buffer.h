#pragma once
#ifndef BUFFER_H
#define BUFFER_H
#include <array>
#include <glew.h>
#include <vector>

enum BufferAccess
{
	StreamDraw  = GL_STREAM_DRAW, 
	StreamRead  = GL_STREAM_READ, 
	StreamCopy  = GL_STREAM_COPY,
	StaticDraw  = GL_STATIC_DRAW, 
	StaticRead  = GL_STATIC_READ, 
	StaticCopy  = GL_STATIC_COPY,
	DynamicDraw = GL_DYNAMIC_DRAW, 
	DynamicRead = GL_DYNAMIC_READ, 
	DynamicCopy = GL_DYNAMIC_COPY
};

enum BufferType
{
	ArrayBuffer       = GL_ARRAY_BUFFER,
	AtomicCounter     = GL_ATOMIC_COUNTER_BUFFER,
	CopyRead          = GL_COPY_READ_BUFFER,
	DispatchIndirect  = GL_DISPATCH_INDIRECT_BUFFER,
	DrawIndirect      = GL_DRAW_INDIRECT_BUFFER,
	ElementArray      = GL_ELEMENT_ARRAY_BUFFER,
	PixelPack         = GL_PIXEL_PACK_BUFFER,
	PixelUnpack       = GL_PIXEL_UNPACK_BUFFER,
	QueryBuffer       = GL_QUERY_BUFFER,
	ShaderStorage     = GL_SHADER_STORAGE_BUFFER,
	TextureBuffer     = GL_TEXTURE_BUFFER,
	TransformFeedback = GL_TRANSFORM_FEEDBACK_BUFFER,
	UniformBuffer     = GL_UNIFORM_BUFFER
};

class Buffer
{
private:
	GLuint buffer;
	BufferType bufferType;
	std::size_t length;
public:
	Buffer();
	Buffer(BufferType type);
	Buffer(Buffer&& other) noexcept;
	~Buffer();

	std::size_t Size() const;

	void CleanUp();
	void Generate(BufferType type, GLsizeiptr size = 0);
	void BindBuffer() const;
	void Reserve(GLsizeiptr size) const;

	template <class T> Buffer(GLenum type, BufferAccess usage, const std::vector<T>& data);
	template<class T> void BufferData(const std::vector<T>& data, BufferAccess usage);
	template<class T, std::size_t i> void BufferData(const std::array<T, i>& data, BufferAccess usage);
	template<template<class, class...> class C, class T, class... Args> void BufferData(const C<T, Args...>& data, BufferAccess usage);
	template<class T> void BufferSubData(const std::vector<T>& data, GLintptr offset);
	template<class T, std::size_t i> void BufferSubData(const std::array<T, i>& data, GLintptr offset);
	template<template<class, class...> class C, class T, class... Args> void BufferSubData(const C<T, Args...>& data, GLintptr offset);
};

template<class T> inline Buffer::Buffer(GLenum type, BufferAccess usage, const std::vector<T>& data) : bufferType(type)
{
	glGenBuffers(1, &this->buffer);
	this->BufferData(data, (GLenum) usage);
}

template<class T> inline void Buffer::BufferData(const std::vector<T>& data, BufferAccess usage)
{
	if (this->buffer)
	{
		glNamedBufferData(this->buffer, data.size() * sizeof(T), data.data(), (GLenum) usage);
		this->length += data.size() * sizeof(T);
	}
}

template<class T, std::size_t i> inline void Buffer::BufferData(const std::array<T, i>& data, BufferAccess usage)
{
	if (this->buffer)
	{
		glBindBuffer(this->bufferType, this->buffer);
		glBufferData(this->bufferType, (GLsizeiptr) i * sizeof(T), data.data(), (GLenum) usage);
		this->length = i * sizeof(T);
	}
}

template<template<class, class...> class C, class T, class... Args> inline void Buffer::BufferData(const C<T, Args...>& data, BufferAccess usage)
{
	if (this->buffer)
	{
		std::vector<T> reserved(std::distance(data.begin(), data.end()));
		for (auto& a : data)
		{
			reserved.push_back(a);
		}
		this->BufferData(reserved, usage);
	}
}

// TODO: Warnings for things
template<class T> inline void Buffer::BufferSubData(const std::vector<T>& data, GLintptr offset)
{
	if (this->buffer)
	{
		std::size_t total = (std::size_t) offset + sizeof(T) * data.size();
		if (total > this->length)
		{
			// WARNING
			return;
		}
		glBufferSubData(this->buffer, offset, (GLsizeiptr) sizeof(T) * data.size(), data.data());
	}
}


template<class T, std::size_t i> inline void Buffer::BufferSubData(const std::array<T, i>& data, GLintptr offset)
{
	if (this->buffer)
	{
		std::size_t total = (std::size_t) offset + sizeof(T) * i;
		if (total > this->length)
		{
			// WARNING
			return;
		}
		glBufferSubData(this->buffer, offset, (GLsizeiptr) sizeof(T) * i, data.data());
	}
}

template<template<class, class...> class C, class T, class... Args> inline void Buffer::BufferSubData(const C<T, Args...>& data, GLintptr offset)
{
	if (this->buffer)
	{
		std::vector<T> reserved(std::distance(data.begin(), data.end()));
		for (auto& a : data)
		{
			reserved.push_back(a);
		}
		this->BufferSubData(reserved, offset);
	}
}

#endif // BUFFER_H
