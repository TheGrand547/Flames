#pragma once
#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H
#include <glew.h>
#include <map>
#include <type_traits>
#include <typeinfo>
#include "Shader.h"
#include "Vertex.h"

class VertexArray
{
protected:
	GLuint array;
public:
	constexpr VertexArray(GLuint array = 0);
	~VertexArray();

	void CleanUp();
	inline void Bind();
	inline void Generate();


	template<class V> void FillArray(Shader& shader);
	template<class T> static void GenerateArrays(T& arrays);
	template<class T> static void GenerateArrays(std::map<T, VertexArray>& arrays);
};

typedef VertexArray VAO;

constexpr VertexArray::VertexArray(GLuint array) : array(array)
{

}

inline void VertexArray::Bind()
{
	glBindVertexArray(this->array);
}

inline void VertexArray::Generate()
{
	this->CleanUp();
	glGenVertexArrays(1, &this->array);
}

template<class V>
inline void VertexArray::FillArray(Shader& shader)
{

}

template<>
inline void VertexArray::FillArray<Vertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*) nullptr);
	glEnableVertexArrayAttrib(this->array, shader.index("vPos"));
}

template<>
inline void VertexArray::FillArray<TextureVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.index("vTex"), 2, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) offsetof(TextureVertex, coordinates));
	glEnableVertexArrayAttrib(this->array, shader.index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.index("vTex"));
}

template<>
inline void VertexArray::FillArray<ColoredVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.index("vColor"), 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), (const void*) offsetof(ColoredVertex, color));
	glEnableVertexArrayAttrib(this->array, shader.index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.index("vColor"));
}

template<>
inline void VertexArray::FillArray<NormalVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(NormalVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.index("vNorm"), 3, GL_FLOAT, GL_FALSE, sizeof(NormalVertex), (const void*) offsetof(NormalVertex, normal));
	glEnableVertexArrayAttrib(this->array, shader.index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.index("vNorm"));
}

template<>
inline void VertexArray::FillArray<MeshVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.index("vNorm"), 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (const void*) offsetof(MeshVertex, normal));
	glVertexAttribPointer(shader.index("vTex"), 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (const void*) offsetof(MeshVertex, texture));
	glEnableVertexArrayAttrib(this->array, shader.index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.index("vNorm"));
	glEnableVertexArrayAttrib(this->array, shader.index("vTex"));
}

template<class T>
static void VertexArray::GenerateArrays(T& arrays)
{
	static_assert(std::is_same<std::remove_reference<decltype(*std::begin(arrays))>::type, VertexArray>::value);
	GLuint *intermediate = new GLuint[std::size(arrays)];
	glGenVertexArrays((GLsizei) std::size(arrays), intermediate);
	for (std::size_t i = 0; i < std::size(arrays); i++)
	{
		arrays[i].array = intermediate[i];
	}
	delete[] intermediate;
}

template<class T>
static void VertexArray::GenerateArrays(std::map<T, VertexArray>& arrays)
{
	GLuint *intermediate = new GLuint[arrays.size()];
	glGenVertexArrays((GLsizei) arrays.size(), intermediate);
	auto begin = std::begin(arrays);
	for (std::size_t i = 0; i < arrays.size(); i++)
	{
		begin->second.array = intermediate[i];
		begin++;
	}
	delete[] intermediate;
}

#endif // VERTEX_ARRAY_H