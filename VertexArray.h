#pragma once
#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H
#include <glew.h>
#include <glm/glm.hpp>
#include <map>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include "Buffer.h"
#include "log.h"
#include "Shader.h"
#include "Vertex.h"
#include "ResourceBank.h"

class VertexArray
{
protected:
	GLuint array;
	GLsizei stride;
	std::map<GLuint, GLsizei> strides;
public:
	VertexArray(GLuint array = 0, GLsizei stride = 0);
	~VertexArray();

	inline GLuint GetArray() const noexcept;

	void CleanUp();
	inline void Bind() const noexcept;
	inline void Generate();
	
	// Assumes this->array is already bound
	inline void BufferBindingPointDivisor(GLuint bindingPoint = 0, GLuint bindingDivisor = 0);

	inline void BindArrayBuffer(const ArrayBuffer& buffer, GLuint bindingPoint = 0, GLintptr offset = 0);
	inline void DoubleBindArrayBuffer(const ArrayBuffer& buffer, GLuint bindingPoint = 0, GLintptr offset = 0);

	template<class V> inline void ArrayFormat(GLuint bindingPoint = 0, GLuint bindingDivisor = 0);
	template<class V> inline void ArrayFormatM(Shader& shader, GLuint bindingPoint = 0, GLuint bindingDivisor = 0, std::string name = "Model");

	template<class V> inline void ArrayFormatOverride(const std::string& name, Shader& shader, GLuint bindingPoint = 0, 
		GLuint bindingDivisor = 0, GLuint relativeOffset = 0, GLsizei stride = 0);

	template<class V> inline void ArrayFormatOverride(const GLuint index, GLuint bindingPoint = 0,
		GLuint bindingDivisor = 0, GLuint relativeOffset = 0, GLsizei stride = 0);

	template<typename T> static void GenerateArrays(T& arrays);
	template<class T> static void GenerateArrays(std::map<T, VertexArray>& arrays);
};

typedef VertexArray VAO;

inline GLuint VertexArray::GetArray() const noexcept
{
	return this->array;
}

inline void VertexArray::Bind() const noexcept
{
	glBindVertexArray(this->array);
}

inline void VertexArray::Generate()
{
	this->CleanUp();
	glGenVertexArrays(1, &this->array);
}

inline void VertexArray::BufferBindingPointDivisor(GLuint bindingPoint, GLuint bindingDivisor)
{
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
}

inline void VertexArray::BindArrayBuffer(const ArrayBuffer& buffer, GLuint bindingPoint, GLintptr offset)
{
	glBindVertexBuffer(bindingPoint, buffer.GetBuffer(), offset, this->strides[bindingPoint]);
}

inline void VertexArray::DoubleBindArrayBuffer(const ArrayBuffer& buffer, GLuint bindingPoint, GLintptr offset)
{
	this->Bind();
	this->BindArrayBuffer(buffer, bindingPoint, offset);
}

// NOTE TO FUTURE READERS, IF ONE OF THESE IS THROWING AN ERROR, SOME VERTEX ATTRIBUTE WAS OPTIMIZED OUT
template<> inline void VertexArray::ArrayFormat<Vertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(0, bindingPoint);
	glEnableVertexAttribArray(0);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(Vertex);
}

template<> inline void VertexArray::ArrayFormat<UIVertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, offsetof(UIVertex, uv));
	glVertexAttribBinding(0, bindingPoint);
	glVertexAttribBinding(1, bindingPoint);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(UIVertex);
}

// This should not be used
/*
template<> inline void VertexArray::ArrayFormat<TangentVertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vTan"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vBtn"), 3, GL_FLOAT, GL_FALSE, offsetof(TangentVertex, biTangent));
	glVertexAttribBinding(shader.Index("vTan"), bindingPoint);
	glVertexAttribBinding(shader.Index("vBtn"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vTan"));
	glEnableVertexAttribArray(shader.Index("vBtn"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(TangentVertex);
}*/

template<> inline void VertexArray::ArrayFormat<TextureVertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, offsetof(TextureVertex, uvs));
	glVertexAttribBinding(0, bindingPoint);
	glVertexAttribBinding(1, bindingPoint);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(TextureVertex);
}

template<> inline void VertexArray::ArrayFormat<ColoredVertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, offsetof(ColoredVertex, color));
	glVertexAttribBinding(0, bindingPoint);
	glVertexAttribBinding(1, bindingPoint);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(ColoredVertex);
}

template<> inline void VertexArray::ArrayFormat<NormalVertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, offsetof(NormalVertex, normal));
	glVertexAttribBinding(0, bindingPoint);
	glVertexAttribBinding(1, bindingPoint);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(NormalVertex);
}

template<> inline void VertexArray::ArrayFormat<MeshVertex>(GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE,  offsetof(MeshVertex, normal));
	glVertexAttribFormat(2, 2, GL_FLOAT, GL_FALSE, offsetof(MeshVertex, texture));
	glVertexAttribBinding(0, bindingPoint);
	glVertexAttribBinding(1, bindingPoint);
	glVertexAttribBinding(2, bindingPoint);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(MeshVertex);
}

// TODO: FIX HACK
template<> inline void VertexArray::ArrayFormatM<glm::mat4>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor, std::string name)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	GLuint index = shader.Index(name);
	for (int i = 0; i < 4; i++)
	{
		glVertexAttribFormat(index + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4) * i);
		glVertexAttribBinding(index + i, bindingPoint);
		glEnableVertexAttribArray(index + i);
	}
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(glm::mat4);
}

template<class V> inline void VertexArray::ArrayFormatOverride(const std::string& name, Shader& shader, 
	GLuint bindingPoint, GLuint bindingDivisor, GLuint relativeOffset, GLsizei stride)
{
	GLuint index = shader.Index(name);
	if (index != -1)
	{
		this->ArrayFormatOverride<V>(index, bindingPoint, bindingDivisor, relativeOffset, stride);
	}
	else
	{
		Log("'{}' has either been optimized out, or does not exist in shader '{}'", name, shader.GetName());
	}
}

template<class V>
inline void VertexArray::ArrayFormatOverride(const GLuint index, GLuint bindingPoint, 
	GLuint bindingDivisor, GLuint relativeOffset, GLsizei stride)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	// TODO: more of these
	if constexpr (std::is_same_v<V, glm::vec1>)
	{
		glVertexAttribFormat(index, 1, GL_FLOAT, GL_FALSE, relativeOffset);
	}
	if constexpr (std::is_same_v<V, glm::vec2>)
	{
		glVertexAttribFormat(index, 2, GL_FLOAT, GL_FALSE, relativeOffset);
	}
	if constexpr (std::is_same_v<V, glm::vec3>)
	{
		glVertexAttribFormat(index, 3, GL_FLOAT, GL_FALSE, relativeOffset);
	}
	if constexpr (std::is_same_v<V, glm::vec4>)
	{
		glVertexAttribFormat(index, 4, GL_FLOAT, GL_FALSE, relativeOffset);
	}
	if constexpr (std::is_same_v<V, glm::mat4>)
	{
		for (int i = 0; i < 4; i++)
		{
			glVertexAttribFormat(index + i, 4, GL_FLOAT, GL_FALSE, relativeOffset + sizeof(glm::vec4) * i);
			glVertexAttribBinding(index + i, bindingPoint);
			glEnableVertexAttribArray(index + i);
		}
	}
	glVertexAttribBinding(index, bindingPoint);
	glEnableVertexAttribArray(index);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	if (!stride)
	{
		this->strides[bindingPoint] += static_cast<GLsizei>(sizeof(V));
	}
	else
	{
		this->strides[bindingPoint] = stride;
	}
}

template<class T> static void VertexArray::GenerateArrays(T& arrays)
{
	static_assert(std::is_same<std::remove_reference<decltype(*std::begin(arrays))>::type, VertexArray>::value);
	GLuint *intermediate = new GLuint[std::size(arrays)];
	glGenVertexArrays(static_cast<GLsizei>(std::size(arrays)), intermediate);
	for (std::size_t i = 0; i < std::size(arrays); i++)
	{
		arrays[i].array = intermediate[i];
	}
	delete[] intermediate;
}

template<class T> static void VertexArray::GenerateArrays(std::map<T, VertexArray>& arrays)
{
	std::vector<GLuint> intermediate(arrays.size());
	glGenVertexArrays(static_cast<GLsizei>(arrays.size()), intermediate.data());
	auto begin = std::begin(arrays);
	for (std::size_t i = 0; i < arrays.size(); i++)
	{
		begin->second.array = intermediate[i];
		begin++;
	}
}
using VAOBank = Bank<VAO>;
#endif // VERTEX_ARRAY_H