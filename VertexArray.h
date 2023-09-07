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

/*
template <typename T>
struct always_false : std::false_type { };

template <typename T>
constexpr bool always_false_v = always_false<T>::value;

template<typename T> concept IsString = std::same_as<T, std::string> || std::same_as<T, const char *>;
template<typename T> concept NotString = !IsString<T>;
template<typename T> concept IsNum = std::is_arithmetic_v<T>;

template<typename T> constexpr std::size_t GetGLSize();
template<std::size_t M, typename base, glm::qualifier alloc, template <std::size_t, typename, glm::qualifier> class T> constexpr std::size_t GetGLSize();
template<typename T> constexpr GLenum GetGLEnum(T t = T());
template<std::size_t M, typename type, glm::qualifier q> constexpr GLenum GetGLEnum(glm::vec<M, type, q> p = glm::vec<M, type, q>());
template<std::size_t M, std::size_t N, typename type, glm::qualifier q> constexpr GLenum GetGLEnum(glm::mat<M, N, type, q> p = glm::mat<M, N, type, q>());
template<typename T, typename W, typename ...types> constexpr std::size_t GetArgumentSize();
template<typename T> constexpr std::size_t GetArgumentSize();


template<IsNum T> constexpr std::size_t GetGLCount(T t = T());
//template<template <std::size_t, class> class T, std::size_t M, class base> constexpr std::size_t GetGLCount();
//template<template <std::size_t, std::size_t, typename> class T, std::size_t M, std::size_t N, typename base> constexpr std::size_t GetGLCount();


template<IsNum T> constexpr std::size_t GetGLCount(T t) { return 1; }
template<std::size_t M, class type, glm::qualifier q> constexpr std::size_t GetGLCount(glm::vec<M, type, q> p = glm::vec<M, type, q>())
{ 
	return M; 
}
//template<template <std::size_t, std::size_t, typename> class T, std::size_t M, std::size_t N> constexpr std::size_t GetGLCount() { return M * N; }


template<class T> constexpr GLenum GetGLEnum(T t) { return GL_INVALID_VALUE; }
template<> constexpr GLenum GetGLEnum<float>(float) { return GL_FLOAT; }
template<> constexpr GLenum GetGLEnum<double>(double) { return GL_DOUBLE; }
template<> constexpr GLenum GetGLEnum<char>(char) { return GL_BYTE; }
template<> constexpr GLenum GetGLEnum<unsigned char>(unsigned char) { return GL_UNSIGNED_BYTE; }
template<> constexpr GLenum GetGLEnum<short>(short) { return GL_SHORT; }
template<> constexpr GLenum GetGLEnum<unsigned short>( unsigned short) { return GL_UNSIGNED_SHORT; }
template<> constexpr GLenum GetGLEnum<int>(int) { return GL_INT; }
template<> constexpr GLenum GetGLEnum<unsigned int>(unsigned int) { return GL_UNSIGNED_INT; }
template<std::size_t M, typename type, glm::qualifier q> constexpr GLenum GetGLEnum(glm::vec<M, type, q> p) { return GetGLEnum(type()); }
template<std::size_t M, std::size_t N, typename type, glm::qualifier q> constexpr GLenum GetGLEnum(glm::mat<M, N, type, q> p) { return GetGLEnum(type()); }
*/

template<GLenum primitive> constexpr std::size_t GetGLPrimativeSize()
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


template<typename T,  typename... types> constexpr std::size_t GetArgumentSize()
{
	return GetGLSize<T>() + GetArgumentSize<types...>();
}

template<typename T> constexpr std::size_t GetArgumentSize() { return GetGLSize<T>(); }


template<typename T> constexpr std::size_t GetGLSize()
{
	return GetGLCount(T()) * GetGLPrimativeSize<GetGLEnum<T>()>();
}


class VertexArray
{
protected:
	GLuint array, bindingPoint;
	GLsizei stride;
	//template<typename T> void FillArrayInternal(Shader& shader, std::size_t stride, std::size_t current, const std::string& first);
public:
	constexpr VertexArray(GLuint array = 0, GLuint bindingPoint = 0, GLsizei stride = 0);
	~VertexArray();

	inline GLuint GetArray() const noexcept;

	void CleanUp();
	inline void BindArrayObject();
	inline void Generate();
	
	inline void BindArrayBuffer(Buffer<ArrayBuffer>& buffer, GLintptr offset = 0);

	template<class V> void FillArray(Shader& shader, GLuint bindingPoint = 0);
	//template<class V> void FillArray2(Shader& shader);
	// TODO: Do this with specified indicies std::size_t
	/*
	template<typename T, typename... types, IsString... Args> void FillArray(Shader& shader, const std::string& first, Args&&... args);
	template<typename T> void FillArray(Shader& shader, const std::string& first);
	template<typename T, typename... types, IsString... Args> void FillArray(Shader& shader, std::size_t size, std::size_t current, const std::string& first, Args&&... args);
	template<typename T> void FillArray(Shader& shader, std::size_t size, std::size_t current);
	*/


	template<typename T> static void GenerateArrays(T& arrays);
	template<class T> static void GenerateArrays(std::map<T, VertexArray>& arrays);
};

typedef VertexArray VAO;

constexpr VertexArray::VertexArray(GLuint array, GLuint bindingPoint, GLsizei stride) : array(array), bindingPoint(bindingPoint), stride(stride)
{

}

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

inline void VertexArray::BindArrayBuffer(Buffer<ArrayBuffer>& buffer, GLintptr offset)
{
	glBindVertexArray(this->array);
	glBindVertexBuffer(this->bindingPoint, buffer.GetBuffer(), offset, this->stride);
}

// TODO: Maybe reinvestigate this later?
/*
template<typename T, typename... types, IsString... Args>
inline void VertexArray::FillArray(Shader& shader, const std::string& first, Args&&... args)
{
	glBindVertexArray(this->array);
	this->FillArrayInternal<T>(shader, GetArgumentSize<T, types...>(), 0, first);
	this->FillArray<types...>(shader, GetArgumentSize<T, types...>(), GetGLSize<T>(), std::forward<Args>(args)...);
}
template<typename T> inline void VertexArray::FillArray(Shader& shader, const std::string& first)
{
	glBindVertexArray(this->array);
	this->FillArrayInternal<T>(shader, GetGLSize<T>(), 0, first);
}
#define STR(X) #X
template<typename T> inline void VertexArray::FillArrayInternal(Shader& shader, std::size_t stride, std::size_t current, const std::string& first)
{
	std::cout << STR(T) << ": " << GetGLEnum<T>() << ": " << GL_FLOAT << ": " << GL_DOUBLE << ": " << GL_INT << ": " << GL_INVALID_VALUE << std::endl;
	glVertexAttribPointer(shader.Index(first), (GLint) GetGLCount(T()), GetGLEnum(T()), GL_FALSE, (GLsizei) stride, (const void*)current);
	CheckError();
	glEnableVertexArrayAttrib(this->array, shader.Index(first));
}


template<typename T, typename... types, IsString... Args>
inline void VertexArray::FillArray(Shader& shader, std::size_t stride, std::size_t current, const std::string& first, Args&&... args)
{
	this->FillArrayInternal<T>(shader, stride, current, first);
	this->FillArray<types..., T>(shader, stride, current + GetGLSize<T>(), std::forward<Args>(args)...); // <- So unbelievably cursed
}

template<typename T> inline void VertexArray::FillArray(Shader& shader, std::size_t strid, std::size_t current) {}*/

// NOTE TO FUTURE READERS, IF ONE OF THESE IS THROWING AN ERROR, SOME VERTEX ATTRIBUTE WAS OPTIMIZED OUT
template<> inline void VertexArray::FillArray<Vertex>(Shader& shader, GLuint bindingPoint)
{
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	this->stride = sizeof(Vertex);
	this->bindingPoint = bindingPoint;
}

template<> inline void VertexArray::FillArray<TextureVertex>(Shader& shader, GLuint bindingPoint)
{
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vTex"), 3, GL_FLOAT, GL_FALSE, offsetof(TextureVertex, uvs));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vTex"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vTex"));
	this->stride = sizeof(TextureVertex);
	this->bindingPoint = bindingPoint;
}

template<> inline void VertexArray::FillArray<ColoredVertex>(Shader& shader, GLuint bindingPoint)
{
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vColor"), 3, GL_FLOAT, GL_FALSE, offsetof(ColoredVertex, color));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vColor"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vColor"));
	this->stride = sizeof(ColoredVertex);
	this->bindingPoint = bindingPoint;
}

template<> inline void VertexArray::FillArray<NormalVertex>(Shader& shader, GLuint bindingPoint)
{
	glBindVertexArray(this->array);
	glBindVertexArray(this->array);
	glVertexAttribFormat(shader.Index("vPos"), 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribFormat(shader.Index("vNorm"), 3, GL_FLOAT, GL_FALSE, offsetof(NormalVertex, normal));
	glVertexAttribBinding(shader.Index("vPos"), bindingPoint);
	glVertexAttribBinding(shader.Index("vNorm"), bindingPoint);
	glEnableVertexAttribArray(shader.Index("vPos"));
	glEnableVertexAttribArray(shader.Index("vNorm"));
	this->stride = sizeof(NormalVertex);
	this->bindingPoint = bindingPoint;
}

template<> inline void VertexArray::FillArray<MeshVertex>(Shader& shader, GLuint bindingPoint)
{
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
	this->stride = sizeof(MeshVertex);
	this->bindingPoint = bindingPoint;
}

template<class T> static void VertexArray::GenerateArrays(T& arrays)
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

template<class T> static void VertexArray::GenerateArrays(std::map<T, VertexArray>& arrays)
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