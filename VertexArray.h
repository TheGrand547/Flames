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
template<std::size_t M, typename base, template  <std::size_t, typename> typename T> GLenum GetGLCount();
template<std::size_t M, std::size_t N, typename base, template <std::size_t, std::size_t, typename> class T> consteval GLenum GetGLCount();
template<typename T> consteval GLenum GetGLEnum();
template<std::size_t M, typename base, template <std::size_t, typename> class T> consteval GLenum GetGLEnum();
template<std::size_t M, std::size_t N, typename base, template <std::size_t, std::size_t, typename> class T> consteval GLenum GetGLEnum();
template<typename T, typename W, typename ...types> consteval std::size_t GetArgumentSize();
template<typename T> consteval std::size_t GetArgumentSize();

template<class T> consteval GLenum GetGLEnum() { return GL_INVALID_VALUE; }
template<> consteval GLenum GetGLEnum<float>() { return GL_FLOAT; }
template<> consteval GLenum GetGLEnum<double>() { return GL_DOUBLE; }
template<> consteval GLenum GetGLEnum<char>() { return GL_BYTE; }
template<> consteval GLenum GetGLEnum<unsigned char>() { return GL_UNSIGNED_BYTE; }
template<> consteval GLenum GetGLEnum<short>() { return GL_SHORT; }
template<> consteval GLenum GetGLEnum<unsigned short>() { return GL_UNSIGNED_SHORT; }
template<> consteval GLenum GetGLEnum<int>() { return GL_INT; }
template<> consteval GLenum GetGLEnum<unsigned int>() { return GL_UNSIGNED_INT; }
template<std::size_t M, typename base, glm::vec<M, base>> consteval GLenum GetGLEnum() { return GetGLEnum<base>(); }
template<std::size_t M, std::size_t N, typename base, glm::mat<M, N, base>> consteval GLenum GetGLEnum() { return GetGLEnum<base>(); }

template<class T> consteval std::size_t GetGLCount() { return 1;  }
template<std::size_t M, typename base, glm::vec<M, base>> consteval GLenum GetGLCount() { return M; }
template<std::size_t M, std::size_t N, typename base, glm::mat<M, N, base>> consteval GLenum GetGLCount() { return M * N; }

template<GLenum primitive> consteval std::size_t GetGLPrimativeSize()
{
	switch (primitive)
	{
	case GL_BYTE: case GL_UNSIGNED_BYTE: return sizeof(char);
	case GL_SHORT: case GL_UNSIGNED_SHORT: return sizeof(short);
	case GL_INT: case GL_UNSIGNED_INT: return sizeof(int);
	case GL_FLOAT: return sizeof(float);
	case GL_DOUBLE: return sizeof(double);
	default: return 0;
	}
}


template<typename T, typename W, typename... types> consteval std::size_t GetArgumentSize()
{
	return GetGLSize<T>() + GetGLSize<W>() + GetArgumentSize<types...>();
}

template<typename T> consteval std::size_t GetArgumentSize() { return GetGLSize<T>(); }

//template<> consteval std::size_t GetArgumentSize<void>() { return 0; }

template<> consteval std::size_t GetGLSize<void>() { return 0; }

template<typename T> consteval std::size_t GetGLSize()
{
	return GetGLCount<T>() * GetGLPrimativeSize<GetGLEnum<T>()>();
}


class VertexArray
{
protected:
	GLuint array;
	template<typename T> void FillArrayInternal(Shader& shader, std::size_t stride, std::size_t current, const std::string& first);
public:
	constexpr VertexArray(GLuint array = 0);
	~VertexArray();

	void CleanUp();
	inline void Bind();
	inline void Generate();


	template<class V> void FillArray(Shader& shader);
	// TODO: Do this with specified indicies std::size_t
	template<typename T, typename... types, IsString... Args> void FillArray(Shader& shader, const std::string& first, Args&&... args);
	template<typename T> void FillArray(Shader& shader, const std::string& first);
	template<typename T, typename... types, IsString... Args> void FillArray(Shader& shader, std::size_t size, std::size_t current, const std::string& first, Args&&... args);
	//template<typename T> void FillArray(Shader& shader, std::size_t size, std::size_t current, const std::string& first);
	template<typename T> static void GenerateArrays(T& arrays);
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


template<typename T, typename... types, IsString... Args>
inline void VertexArray::FillArray(Shader& shader, const std::string& first,  Args&&... args)
{
	glBindVertexArray(this->array);
	this->FillArrayInternal<T>(shader, GetArgumentSize<types...>(), 0, first);
	this->FillArray<T, types..., Args...>(shader, GetArgumentSize<types...>(), GetGLSize<T>(), std::forward<Args>(args)...);
}

template<typename T> inline void VertexArray::FillArray(Shader& shader, const std::string& first)
{
	glBindVertexArray(this->array);
	this->FillArrayInternal<T>(shader, GetGLSize<T>(), 0, first);
}


template<typename T> inline void VertexArray::FillArrayInternal(Shader& shader, std::size_t stride, std::size_t current, const std::string& first)
{
	glVertexAttribPointer(shader.Index(first), GetGLCount<T>(), GetGLEnum<T>(), GL_FALSE, (GLsizei) stride, (const void*)current);
	glEnableVertexArrayAttrib(this->array, shader.Index(first));
}


template<typename T, typename... types, IsString... Args>
inline void VertexArray::FillArray(Shader& shader, std::size_t stride, std::size_t current, const std::string& first, Args&&... args)
{
	/*
	glVertexAttribPointer(shader.Index(first), GetGLCount<T>(), GetGLEnum<T>(), GL_FALSE, (GLsizei) stride, (const void*) current);
	glEnableVertexArrayAttrib(this->array, shader.Index(first));*/
	this->FillArrayInternal<T>(shader, stride, current, first);
	this->FillArray<types...>(shader, stride, current + GetGLSize<T>(), std::forward<Args>(args)...);
}

/*
template<class T>
inline void VertexArray::FillArray(Shader& shader, std::size_t stride, std::size_t current, const std::string& first)
{
	this->FillArrayInternal<T>(shader, stride, current + GetGLSize<T>(), first);
}*/


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