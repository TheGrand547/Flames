#pragma once
#ifndef VERTEX_ARRAY_H
#define VERTEX_ARRAY_H
#include <glew.h>
#include <glm/glm.hpp>
#include <map>
#include <type_traits>
#include <typeinfo>
#include <vector>
#include "Shader.h"
#include "Vertex.h"

template<typename T> concept IsString = std::convertible_to<T, std::string>;

template<typename T> consteval std::size_t GetGLSize();
template<typename T> consteval std::size_t GetGLCount();
template<std::size_t M, typename base, template <std::size_t, typename> typename T> GLenum GetGLCount();
template<std::size_t M, std::size_t N, typename base, template <std::size_t, std::size_t, typename> class T> GLenum GetGLCount();
template<typename T> consteval GLenum GetGLEnum();
template<std::size_t M, typename base, template <std::size_t, typename> class T> GLenum GetGLEnum();
template<std::size_t M, typename base, template <std::size_t, typename> class T> GLenum GetGLEnum();
template<std::size_t M, std::size_t N, typename base, template <std::size_t, std::size_t, typename> class T> GLenum GetGLEnum();
template<class T, typename ...types> consteval std::size_t GetArgumentSize();

template<> consteval GLenum GetGLEnum<float>() { return GL_FLOAT; }
template<> consteval GLenum GetGLEnum<double>() { return GL_DOUBLE; }
template<> consteval GLenum GetGLEnum<char>() { return GL_BYTE; }
template<> consteval GLenum GetGLEnum<unsigned char>() { return GL_UNSIGNED_BYTE; }
template<> consteval GLenum GetGLEnum<short>() { return GL_SHORT; }
template<> consteval GLenum GetGLEnum<unsigned short>() { return GL_UNSIGNED_SHORT; }
template<> consteval GLenum GetGLEnum<int>() { return GL_INT; }
template<> consteval GLenum GetGLEnum<unsigned int>() { return GL_UNSIGNED_INT; }
//template<std::size_t M, typename base> GLenum GetGLEnum<glm::vec<M, base>>() { return GetGLEnum<base>(); }
template<std::size_t M, std::size_t N, typename base> GLenum GetGLEnum<glm::mat<M, N, base>>() { return GetGLEnum<base>(); }
template<std::size_t M, typename base> GLenum GetGLCount<glm::vec<M, base>>() { return M; }
template<std::size_t M, std::size_t N, typename base> GLenum GetGLCount<glm::mat<M, N, base>>() { return M * N; }


template<class T, typename ...types> consteval std::size_t GetArgumentSize()
{
	return GetGLSize<T>() + GetArgumentSize<types...>();
}

template<typename T> consteval std::size_t GetGLSize()
{
	//return GetGLCount<T> * 
}

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
	template<class T, typename ...types, IsString... Args> void FillArray(Shader& shader, const std::string& first, const Args&... args);
	template<class T, typename ...types, IsString... Args> void FillArray(Shader& shader, std::size_t size, const std::string& first, const Args&... args);
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


template<class T, typename ...types, IsString ...Args>
inline void VertexArray::FillArray(Shader& shader, const std::string& first, const Args& ...args)
{
	glBindVertexArray(this->array);
	this->FillArray<T, types..., Args...>(shader, GetArgumentSize<types...>(), first, std::forward<Args>(args)...);
}

template<class T, typename ...types, IsString ...Args>
inline void VertexArray::FillArray(Shader& shader, std::size_t stride, const std::string& first, const Args& ...args)
{
	glVertexAttribPointer(shader.Index(first), GetGLSize<T>(), GetGLEnum<T>(), GL_FALSE, stride, (const void*)GetGLSize<T>());
	glEnableVertexArrayAttrib(this->array, shader.Index(first));
	this->FillArray<T, types..., Args...>(shader, stride, std::forward<Args>(args)...);
}


template<>
inline void VertexArray::FillArray<Vertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*) nullptr);
	glEnableVertexArrayAttrib(this->array, shader.Index("vPos"));
}

template<>
inline void VertexArray::FillArray<TextureVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.Index("vTex"), 2, GL_FLOAT, GL_FALSE, sizeof(TextureVertex), (const void*) offsetof(TextureVertex, coordinates));
	glEnableVertexArrayAttrib(this->array, shader.Index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.Index("vTex"));
}

template<>
inline void VertexArray::FillArray<ColoredVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.Index("vColor"), 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), (const void*) offsetof(ColoredVertex, color));
	glEnableVertexArrayAttrib(this->array, shader.Index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.Index("vColor"));
}

template<>
inline void VertexArray::FillArray<NormalVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(NormalVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.Index("vNorm"), 3, GL_FLOAT, GL_FALSE, sizeof(NormalVertex), (const void*) offsetof(NormalVertex, normal));
	glEnableVertexArrayAttrib(this->array, shader.Index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.Index("vNorm"));
}

template<>
inline void VertexArray::FillArray<MeshVertex>(Shader& shader)
{
	glBindVertexArray(this->array);
	glVertexAttribPointer(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (const void*) nullptr);
	glVertexAttribPointer(shader.Index("vNorm"), 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (const void*) offsetof(MeshVertex, normal));
	glVertexAttribPointer(shader.Index("vTex"), 2, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), (const void*) offsetof(MeshVertex, texture));
	glEnableVertexArrayAttrib(this->array, shader.Index("vPos"));
	glEnableVertexArrayAttrib(this->array, shader.Index("vNorm"));
	glEnableVertexArrayAttrib(this->array, shader.Index("vTex"));
}

template<class T>
static void VertexArray::GenerateArrays(T& arrays)
{
	static_assert(std::is_same<std::remove_reference<decltype(*std::begin(arrays))>::type, VertexArray>::value);
	GLuint *intermediate = (GLuint) new GLuint[std::size(arrays)];
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
	std::vector<GLuint> intermediate(arrays.size());
	glGenVertexArrays((GLsizei) arrays.size(), intermediate.data());
	auto begin = std::begin(arrays);
	for (std::size_t i = 0; i < arrays.size(); i++)
	{
		begin->second.array = intermediate[i];
		begin++;
	}
}

#endif // VERTEX_ARRAY_H