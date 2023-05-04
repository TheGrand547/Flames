#pragma once
#ifndef BUFFER_H
#define BUFFER_H
#include <glew.h>
#include <array>
#include <vector>

class Buffer
{
private:
	GLuint buffer;
	GLenum bufferType;
	size_t length;
public:
	Buffer();
	Buffer(GLenum type);
	Buffer(Buffer&& other) noexcept;
	~Buffer();

	size_t Size() const;

	void CleanUp();
	void Generate(GLenum type);
	void BindBuffer() const;
	void Reserve(GLsizeiptr size) const;

	template <class T> Buffer(GLenum type, GLenum usage, const std::vector<T>& data);
	template<class T> void BufferData(const std::vector<T>& data, GLenum usage);
	template<class T, int i> void BufferData(const std::array<T, i>& data, GLenum usage);
	template<template<class, class...> class C, class T, class... Args> void BufferData(const C<T, Args...>& data, GLenum usage);
	template<class T> void BufferSubData(const std::vector<T>& data, GLintptr offset);
	template<class T, int i> void BufferSubData(const std::array<T, i>& data, GLintptr offset);
	template<template<class, class...> class C, class T, class... Args> void BufferSubData(const C<T, Args...>& data, GLintptr offset);
};

template<class T> inline Buffer::Buffer(GLenum type, GLenum usage, const std::vector<T>& data) : bufferType(type)
{
	glGenBuffers(1, &this->buffer);
	this->BufferData(data, usage);
}

template<class T> inline void Buffer::BufferData(const std::vector<T>& data, GLenum usage)
{
	if (this->buffer)
	{
		glNamedBufferData(this->buffer, data.size() * sizeof(T), data.data(), usage);
		this->length += data.size() * sizeof(T);
	}
}

template<class T, int i> inline void Buffer::BufferData(const std::array<T, i>& data, GLenum usage)
{
	if (this->buffer)
	{
		glBindBuffer(this->bufferType, this->buffer);
		glBufferData(this->bufferType, i * sizeof(T), data.data(), usage);
		this->length += i * sizeof(T);
	}
}

template<template<class, class...> class C, class T, class... Args> inline void BufferData(const C<T, Args...>& data, GLenum usage)
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

template<class T> inline void Buffer::BufferSubData(const std::vector<T>& data, GLintptr offset)
{
	if (this->buffer)
	{
		//glBufferSubData(this->buffer, )
		//glNamedBufferData(this->buffer, data.size() * sizeof(T), data.data(), usage);
		this->length += data.size() * sizeof(T);
	}
}


template<class T, int i> inline void Buffer::BufferSubData(const std::array<T, i>& data, GLintptr offset)
{
	if (this->buffer)
	{
		//glBufferSubData()
		//glNamedBufferData(this->buffer, i * sizeof(T), data.data(), usage);
		this->length += i * sizeof(T);
	}
}

template<template<class, class...> class C, class T, class... Args> inline void BufferSubData(const C<T, Args...>& data, GLintptr offset)
{
	if (this->buffer)
	{
		std::vector<T> reserved(std::distance(data.begin(), data.end()));
		for (auto& a : data)
		{
			reserved.push_back(a);
		}
		//this->BufferSubData(reserved, usage);
	}
}

#endif // BUFFER_H
