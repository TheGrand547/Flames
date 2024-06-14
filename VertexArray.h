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
	inline void BindArrayObject();
	inline void Generate();
	
	// Assumes this->array is already bound
	inline void BufferBindingPointDivisor(GLuint bindingPoint = 0, GLuint bindingDivisor = 0);

	inline void BindArrayBuffer(ArrayBuffer& buffer, GLuint bindingPoint = 0, GLintptr offset = 0);

	template<class V> inline void ArrayFormat(Shader& shader, GLuint bindingPoint = 0, GLuint bindingDivisor = 0);

	template<class V> inline void ArrayFormatOverride(const std::string& name, Shader& shader, GLuint bindingPoint = 0, 
		GLuint bindingDivisor = 0, GLuint relativeOffset = 0);

	template<typename T> static void GenerateArrays(T& arrays);
	template<class T> static void GenerateArrays(std::map<T, VertexArray>& arrays);
};

typedef VertexArray VAO;

inline GLuint VertexArray::GetArray() const noexcept
{
	return this->array;
}

inline void VertexArray::BindArrayObject()
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

inline void VertexArray::BindArrayBuffer(ArrayBuffer& buffer, GLuint bindingPoint, GLintptr offset)
{
	glBindVertexArray(this->array);
	glBindVertexBuffer(bindingPoint, buffer.GetBuffer(), offset, this->strides[bindingPoint]);
}

// NOTE TO FUTURE READERS, IF ONE OF THESE IS THROWING AN ERROR, SOME VERTEX ATTRIBUTE WAS OPTIMIZED OUT
template<> inline void VertexArray::ArrayFormat<Vertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(Vertex);
}

template<> inline void VertexArray::ArrayFormat<UIVertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 2, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vTex"), 2, GL_FLOAT, GL_FALSE, offsetof(UIVertex, uv));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vTex"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vTex"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(UIVertex);
}

template<> inline void VertexArray::ArrayFormat<TangentVertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
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
}

template<> inline void VertexArray::ArrayFormat<TextureVertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vTex"), 2, GL_FLOAT, GL_FALSE, offsetof(TextureVertex, uvs));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vTex"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vTex"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(TextureVertex);
}

template<> inline void VertexArray::ArrayFormat<ColoredVertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vColor"), 3, GL_FLOAT, GL_FALSE, offsetof(ColoredVertex, color));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vColor"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vColor"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(ColoredVertex);
}

template<> inline void VertexArray::ArrayFormat<NormalVertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vNorm"), 3, GL_FLOAT, GL_FALSE, offsetof(NormalVertex, normal));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vNorm"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vNorm"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(NormalVertex);
}

template<> inline void VertexArray::ArrayFormat<MeshVertex>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vNorm"), 3, GL_FLOAT, GL_FALSE,  offsetof(MeshVertex, normal));
	glVertexAttribFormat(shader.Index("vTex"), 2, GL_FLOAT, GL_FALSE, offsetof(MeshVertex, texture));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vNorm"), bindingPoint);
	glVertexAttribBinding(shader.Index("vTex"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vNorm"));
	glEnableVertexAttribArray(shader.Index("vTex"));
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(MeshVertex);
}

template<> inline void VertexArray::ArrayFormat<glm::mat4>(Shader& shader, GLuint bindingPoint, GLuint bindingDivisor)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	GLuint index = shader.Index("Model");
	for (int i = 0; i < 4; i++)
	{
		glVertexAttribFormat(index + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4) * i);
		glVertexAttribBinding(index + i, bindingPoint);
		glEnableVertexAttribArray(index + i);
	}
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	this->strides[bindingPoint] = sizeof(glm::mat4);
}

template<class V> inline void VertexArray::ArrayFormatOverride(const std::string& name, Shader& shader, GLuint bindingPoint, GLuint bindingDivisor, GLuint relativeOffset)
{
	if (!this->array) this->Generate();
	glBindVertexArray(this->array);
	// TODO: more of these
	GLuint index = shader.Index(name);
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
	glVertexAttribBinding(index, bindingPoint);
	glEnableVertexAttribArray(index);
	glVertexBindingDivisor(bindingPoint, bindingDivisor);
	if (!relativeOffset)
	{
		this->strides[bindingPoint] = sizeof(V);
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

#endif // VERTEX_ARRAY_H