#pragma once
#ifndef BUFFER_H
#define BUFFER_H
#include <array>
#include <glew.h>
#include <map>
#include <span>
#include <vector>
#include "log.h"

// TODO: Paranoia checks removal in release, ie if RELEASE then don't do a bunch of checks with error messages

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
	ArrayBuffer         = GL_ARRAY_BUFFER,
	AtomicCounter       = GL_ATOMIC_COUNTER_BUFFER,
	CopyRead            = GL_COPY_READ_BUFFER,
	DispatchIndirect    = GL_DISPATCH_INDIRECT_BUFFER,
	DrawIndirect        = GL_DRAW_INDIRECT_BUFFER,
	ElementArray        = GL_ELEMENT_ARRAY_BUFFER,
	PixelPack           = GL_PIXEL_PACK_BUFFER,
	PixelUnpack         = GL_PIXEL_UNPACK_BUFFER,
	QueryBuffer         = GL_QUERY_BUFFER,
	ShaderStorage       = GL_SHADER_STORAGE_BUFFER,
	TextureBuffer       = GL_TEXTURE_BUFFER,
	TransformFeedback   = GL_TRANSFORM_FEEDBACK_BUFFER,
	UniformBufferObject = GL_UNIFORM_BUFFER
};

template<BufferType Type>
class Buffer
{
protected:
	GLuint buffer;
	std::size_t length;
	// TODO: Maybe union?
	GLsizei elementCount;
	GLenum elementType; // Sloppy, but only for ElementArrayBuffer
	std::size_t elementSize;
public:
	Buffer();
	Buffer(Buffer<Type>&& other) noexcept;
	~Buffer();

	inline GLuint GetBuffer() const;
	inline GLuint GetElementCount() const;
	inline GLenum GetElementType() const;
	inline std::size_t GetElementSize() const;

	std::size_t Size() const;

	void CleanUp();
	void Generate(BufferAccess access = StaticDraw, GLsizeiptr size = 0);
	void BindBuffer() const;
	void Reserve(BufferAccess access, GLsizeiptr size);

	// Constructor with buffer creation
	template<class T> Buffer(const T& data, BufferAccess usage = StaticDraw);
	Buffer(std::size_t size, BufferAccess usage = StaticDraw);

	void Reserve(std::size_t size, BufferAccess usage = StaticDraw);

	// Generic contiguous data structure data
	template<class T> void BufferData(std::span<const T> data, BufferAccess usage = StaticDraw);
	template<class T> void BufferData(std::span<T> data, BufferAccess usage = StaticDraw);

	// Convenience for the easy data structures
	template<class T, std::size_t i> void BufferData(const std::array<T, i>& data, BufferAccess usage = StaticDraw); 
	template<class T> void BufferData(const std::vector<T>& data, BufferAccess usage = StaticDraw);

	// Generic data structure buffer data
	template<template<class, class...> class C, class T, class... Args> void BufferData(const C<T, Args...>& data, BufferAccess usage = StaticDraw);
	
	// For inserting a single element
	template<class T> void BufferSubData(const T& data, GLintptr offset = 0);
	
	// Convenience for easy data structures
	template<class T> void BufferSubData(const std::vector<T>& data, GLintptr offset = 0);
	template<class T, std::size_t i> void BufferSubData(const std::array<T, i>& data, GLintptr offset = 0);

	// Generic contiguous access templates
	template<class T> void BufferSubData(std::span<const T> data, GLintptr offset = 0);
	template<class T> void BufferSubData(std::span<T> data, GLintptr offset = 0);

	// Generic data structure template
	template<template<class, class...> class C, class T, class... Args> void BufferSubData(const C<T, Args...>& data, GLintptr offset = 0);

	template<class T> static void GenerateBuffers(T& buffers);
	template<class T> static void GenerateBuffers(std::map<T, Buffer>& buffers);
};

template<BufferType Type> inline Buffer<Type>::Buffer() : buffer(0), length(0), elementCount(0), elementType(0), elementSize(0)
{

}

template<BufferType Type> inline Buffer<Type>::Buffer(std::size_t size, BufferAccess usage) : buffer(0), length(0), elementCount(0), elementType(0), elementSize(0)
{
	this->Reserve(size, usage);
}

template<BufferType Type> inline Buffer<Type>::Buffer(Buffer<Type>&& other) noexcept 
{
	this->CleanUp();
	this->buffer = other.buffer;
	this->length = other.length;
	this->elementCount = other.elementCount;
	this->elementType = other.elementType;
	this->elementSize = other.elementSize;
	other.buffer = 0;
	other.CleanUp();
}

template<BufferType Type> template<class T> inline Buffer<Type>::Buffer(const T& data, BufferAccess usage)
{
	glGenBuffers(1, &this->buffer);
	this->BufferData(data, usage);
}

template<BufferType Type> inline Buffer<Type>::~Buffer()
{
	this->CleanUp();
}

template<BufferType Type> inline GLuint Buffer<Type>::GetBuffer() const
{
	return this->buffer;
}

template<BufferType Type> inline GLuint Buffer<Type>::GetElementCount() const
{
	return this->elementCount;
}

template<BufferType Type> inline std::size_t Buffer<Type>::GetElementSize() const
{
	return this->elementSize;
}

template<BufferType Type> inline GLenum Buffer<Type>::GetElementType() const
{
	return this->elementType;
}

template<BufferType Type> inline std::size_t Buffer<Type>::Size() const
{
	return this->length;
}

template<BufferType Type> inline void Buffer<Type>::CleanUp()
{
	if (this->buffer)
	{
		glDeleteBuffers(1, &this->buffer);
	}
	this->buffer = 0;
	this->length = 0;
	this->elementCount = 0;
	this->elementSize = 0;
	this->elementType = GL_UNSIGNED_INT;
}

template<BufferType Type> void Buffer<Type>::Generate(BufferAccess access, GLsizeiptr size)
{
	this->CleanUp();
	glGenBuffers(1, &this->buffer);
	if (size)
	{
		this->Reserve(access, size);
	}
}

template<BufferType Type> inline void Buffer<Type>::BindBuffer() const
{
	glBindBuffer((GLenum) Type, this->buffer);
}

template<BufferType Type> inline void Buffer<Type>::Reserve(BufferAccess access, GLsizeiptr size)
{
	if (this->buffer)
	{
		glNamedBufferData(this->buffer, size, nullptr, (GLenum) access);
		this->length = size;
	}
}

template<BufferType Type> inline void Buffer<Type>::Reserve(std::size_t size, BufferAccess usage)
{
	this->Generate();
	glBindBuffer(Type, this->buffer);
	glBufferData(Type, (GLsizei) size, nullptr, (GLenum) usage);
	this->length = size;
	this->elementCount = 0;
	this->elementSize = 0;
	this->elementSize = GL_UNSIGNED_INT;
}

template<BufferType Type> template<class T> inline void Buffer<Type>::BufferData(std::span<T> data, BufferAccess usage)
{
	this->BufferData(std::span<const T>(data), usage);
}

template<BufferType Type> template<class T> inline void Buffer<Type>::BufferData(std::span<const T> data, BufferAccess usage)
{
	if (this->buffer)
	{
		glBindBuffer(Type, this->buffer);
		glBufferData(Type, data.size() * sizeof(T), data.data(), (GLenum) usage);
		this->length = data.size() * sizeof(T);
		this->elementCount = (GLsizei) data.size();
		this->elementType = (sizeof(T) == 1) ? GL_UNSIGNED_BYTE : ((sizeof(T) == 2) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT);
		this->elementSize = sizeof(T);
	}
}


template<BufferType Type> template<class T, std::size_t i> inline void Buffer<Type>::BufferData(const std::array<T, i>& data, BufferAccess usage)
{
	if (this->buffer)
	{
		glBindBuffer(Type, this->buffer);
		glBufferData(Type, (GLsizeiptr) i * sizeof(T), data.data(), (GLenum) usage);
		this->length = i * sizeof(T);
		this->elementCount = (GLsizei) data.size();
		this->elementType = (sizeof(T) == 1) ? GL_UNSIGNED_BYTE : ((sizeof(T) == 2) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT);
		this->elementSize = sizeof(T);
	}
}

template<BufferType Type> template<class T> void Buffer<Type>::BufferData(const std::vector<T>& data, BufferAccess usage)
{
	this->BufferData(std::span<const T>(data), usage);
}

template<BufferType Type> template<template<class, class...> class C, class T, class... Args> inline void Buffer<Type>::BufferData(const C<T, Args...>& data, BufferAccess usage)
{
	if (this->buffer)
	{
		std::vector<T> reserved{};
		reserved.reserve(std::distance(data.begin(), data.end()));
		for (auto& a : data)
		{
			reserved.push_back(a);
		}
		this->BufferData(std::span<const T>(reserved), usage);
	}
}

template<BufferType Type> template<class T> inline void Buffer<Type>::BufferSubData(const T& data, GLintptr offset)
{
	if (this->buffer)
	{
		glBindBuffer(Type, this->buffer);
		glBufferSubData(Type, offset, (GLsizeiptr) sizeof(T), &data);
	}
}

template<BufferType Type> template<class T> inline void Buffer<Type>::BufferSubData(std::span<T> data, GLintptr offset)
{
	this->BufferSubData(std::span<const T>(data), offset);
}

template<BufferType Type> template<class T, std::size_t i> void Buffer<Type>::BufferSubData(const std::array<T, i>& data, GLintptr offset)
{
	this->BufferSubData(std::span<const T>(data), offset);
}

template<BufferType Type> template<class T> void Buffer<Type>::BufferSubData(const std::vector<T>& data, GLintptr offset)
{
	this->BufferSubData(std::span<const T>(data), offset);
}

template<BufferType Type> template<class T> inline void Buffer<Type>::BufferSubData(std::span<const T> data, GLintptr offset)
{
	if (this->buffer)
	{
		std::size_t total = (std::size_t) offset + sizeof(T) * data.size();
		if (total > this->length)
		{
			LogF("Attemptign to write up to memory %zu, but buffer is only %zu long.\n", total, this->length);
			return;
		}
		glBindBuffer(Type, this->buffer);
		glBufferSubData(Type, offset, (GLsizeiptr) sizeof(T) * data.size(), data.data());
	}
}

template<BufferType Type> template<template<class, class...> class C, class T, class... Args> inline void Buffer<Type>::BufferSubData(const C<T, Args...>& data, GLintptr offset)
{
	if (this->buffer)
	{
		std::vector<T> reserved{};
		reserved.reserve(std::distance(data.begin(), data.end()));
		for (auto& a : data)
		{
			reserved.push_back(a);
		}
		this->BufferSubData(std::span<const T>(reserved), offset);
	}
}

template<BufferType Type> template<class T> inline void Buffer<Type>::GenerateBuffers(T& buffers)
{
	static_assert(std::is_same<std::remove_reference<decltype(*std::begin(buffers))>::type, Buffer<Type>>::value);
	GLuint* intermediate = new GLuint[std::size(buffers)];
	glGenBuffers((GLsizei) std::size(buffers), intermediate);
	for (std::size_t i = 0; i < std::size(buffers); i++)
	{
		buffers[i].buffer = intermediate[i];
	}
	delete[] intermediate;
}

template<BufferType Type> template<class T> inline void Buffer<Type>::GenerateBuffers(std::map<T, Buffer<Type>>& buffers)
{
	GLuint* intermediate = new GLuint[buffers.size()];
	glGenBuffers((GLsizei) buffers.size(), intermediate);
	auto begin = std::begin(buffers);
	for (std::size_t i = 0; i < buffers.size(); i++)
	{
		begin->buffer = intermediate[i];
		begin++;
	}
	delete[] intermediate;
}

#endif // BUFFER_H
