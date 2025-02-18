#pragma once
#ifndef FLAMES_SHADER_H
#define FLAMES_SHADER_H
#include <bit>
#include <filesystem>
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include "Buffer.h"
#include "Texture2D.h"
#include "CubeMap.h"
#include "DrawStruct.h"

enum struct PrimitiveDrawingType : unsigned int
{
	Lines                  = GL_LINES,
	LinesAdjacency         = GL_LINES_ADJACENCY,          // Only with Geometry Shaders
	LineLoop               = GL_LINE_LOOP,
	LineStrip              = GL_LINE_STRIP,
	LineStripAdjacency     = GL_LINE_STRIP_ADJACENCY,     // Only with Geometry Shaders
	Patches                = GL_PATCHES,                  // Only with Tesselation Shaders
	Points                 = GL_POINTS,
	Triangle               = GL_TRIANGLES,
	TriangleAdjacency      = GL_TRIANGLES_ADJACENCY,      // Only with Geometry Shaders
	TriangleFan            = GL_TRIANGLE_FAN,
	TriangleStrip          = GL_TRIANGLE_STRIP,
	TriangleStripAdjacency = GL_TRIANGLE_STRIP_ADJACENCY, // Only with Geometry Shaders
};

using DrawType = PrimitiveDrawingType;

// Compute shaders will be their own thing

enum ShaderStages : unsigned char
{
	None                   = 0,
	VertexOnly             = 1,
	Geometry               = 2,
	Tesselation            = 4,
	GeometryAndTesselation = Geometry | Tesselation,
};

struct IndirectDraw
{
	GLuint count, primCount, first, baseInstance;
};

class Shader
{
protected:
	bool compiled, precompiled;
	GLuint program;
	ShaderStages stages;

	std::string name;

	// True -> loaded successfully from file, False -> did not load shader from file
	bool TryLoadCompiled(const std::string& name, std::chrono::system_clock::rep threshold);

	// True -> Compiled Fine, False -> Some Error
	bool ProgramStatus();
public:
	Shader(ShaderStages stages = None);
	Shader(const std::string& name);
	Shader(const std::string& vertex, const std::string& fragment);
	Shader(const char* vertex, const char* fragment);
	Shader(const Shader& other) = delete;
	Shader(Shader&& other) noexcept;
	~Shader();

	Shader& operator=(const Shader& other) = delete;
	Shader& operator=(Shader&& other) noexcept;

	// Compiles all stages available starting with the given filename
	// [name]v, [name]f,  [name]g,            [name]tc,             [name]te 
	// Vertex, fragment, geometry, tesselation control, tesselation evaluate, respectively
	bool CompileSimple(const std::string& name);


	bool Compile(const std::string& vertex, const std::string& frag);

	// These all work on shader code as a string, not read in from a file
	bool CompileEmbedded(const std::string& vertex, const std::string& fragment);
	bool CompileEmbeddedGeometry(const std::string& vertex, const std::string& fragment, const std::string& geometry);
	bool CompileEmbeddedGeometryTesselation(const std::string& vertex, const std::string& fragment, const std::string& geometry, 
		const std::string& tessControl, const std::string& tessEval);
	bool CompileEmbeddedTesselation(const std::string& vertex, const std::string& fragment, const std::string& tessControl, const std::string& tessEval);

	constexpr bool Compiled() const;

	GLuint Index(const std::string& name) const;
	GLuint UniformIndex(const std::string& name) const;
	GLuint UniformBlockIndex(const std::string& name) const;

	void CleanUp();
	void ExportCompiled() const;

	inline GLuint GetProgram() const noexcept;
	inline void SetActiveShader() const noexcept;
	inline void SetInt(const std::string& name, const int i) const noexcept;
	inline void SetUnsignedInt(const std::string& name, const unsigned int i) const noexcept;
	inline void SetFloat(const std::string& name, const float i) const noexcept;
	inline void SetVec2(const std::string& name, const glm::vec2& vec) const noexcept;
	inline void SetVec3(const std::string& name, const glm::vec3& vec) const noexcept;
	inline void SetVec4(const std::string& name, const glm::vec4& vec) const noexcept;
	inline void SetMat4(const std::string& name, const glm::mat4& mat) const noexcept;
	inline void SetMat4s(const std::string& name, const std::span<glm::mat4> mats) const noexcept;
	inline void SetTextureUnit(const std::string& name, const GLuint unit = 0) const noexcept;
	inline void SetTextureUnit(const std::string& name, const Texture2D& texture, const GLuint unit = 0) const noexcept;
	inline void SetTextureUnit(const std::string& name, const CubeMap& texture, const GLuint unit = 0) const noexcept;
	inline void UniformBlockBinding(const std::string& name, GLuint bindingPoint) const noexcept;

	// TODO: Maybe a draw function that takes a variable amount of VAO, Buffers, etc and does all that in a "magic" step

	// Feed the data in the buffer, unchanged, directly to the shader
	inline void DrawArray(PrimitiveDrawingType type, const GLuint primitiveCount, const GLuint elementOffset = 0);
	inline void DrawArray(PrimitiveDrawingType type, ArrayBuffer& buffer, const GLuint elementOffset = 0);
	template<PrimitiveDrawingType type = DrawType::Triangle> inline void DrawArray(const GLuint primitiveCount, const GLuint elementOffset = 0);
	template<PrimitiveDrawingType type = DrawType::Triangle> inline void DrawArray(ArrayBuffer& buffer, const GLuint elementOffset = 0);

	// Feed the shader the data in the buffer, augmented by the instance number, instance count times
	template<PrimitiveDrawingType type = DrawType::Triangle> inline void DrawArrayInstanced(const GLuint primitiveCount, const GLuint instanceCount, 
																		const GLuint primitiveOffset = 0, const GLuint instanceOffset = 0);
	template<PrimitiveDrawingType type = DrawType::Triangle> inline void DrawArrayInstanced(ArrayBuffer& primitiveBuffer, ArrayBuffer& instanceBuffer,
																		const GLuint elementOffset = 0, const GLuint instanceOffset = 0);

	// Draw instanced data from the given parameter struct
	template<PrimitiveDrawingType type = DrawType::Triangle> inline void DrawArrayIndirect(const IndirectDraw& parameters);

	// TODO: maybe glDrawElementsRange() but probably not

	// Feed the data in the buffer to the shader in the order given by the passed container
	template<PrimitiveDrawingType type = DrawType::Triangle, class Container> inline void DrawElementsMemory(const Container& contents);
	template<class Container> inline void DrawElementsMemory(PrimitiveDrawingType type, const Container& contents);

	// Feed the data in the buffer to the shader in the order given by the ElementArray buffer
	template<PrimitiveDrawingType type = DrawType::Triangle> inline void DrawElements(ElementArray& indexBuffer, const GLuint elementOffset = 0, const GLuint indexOffset = 0);
	inline void DrawElements(PrimitiveDrawingType type, ElementArray& indexBuffer, const GLuint elementOffset = 0, const GLuint indexOffset = 0);

	template<PrimitiveDrawingType type> inline void DrawElementsInstanced(ElementArray& indexBuffer, ArrayBuffer& instanceBuffer,
															const GLuint elementOffset = 0, const GLuint indexOffset = 0, const GLint instanceOffset = 0);

	template<PrimitiveDrawingType type = DrawType::Triangle> void DrawElements(const Buffer<BufferType::DrawIndirect>& indirect, const GLuint offset = 0) const noexcept;
	template<PrimitiveDrawingType type = DrawType::Triangle> void MultiDrawElements(const Buffer<BufferType::DrawIndirect>& indirect) const noexcept;
	template<PrimitiveDrawingType type = DrawType::Triangle> void MultiDrawElements(const Buffer<BufferType::DrawIndirect>& indirect,
		const GLuint count) const noexcept;
	template<PrimitiveDrawingType type = DrawType::Triangle> void MultiDrawElements(const Buffer<BufferType::DrawIndirect>& indirect,
		const GLuint offset, const GLuint count) const noexcept;

	static void IncludeInShaderFilesystem(const std::string& virtualName, const std::string& fileName);
	static void SetBasePath(const std::string& basePath);
	static void SetRecompilationFlag(bool flag);
};

constexpr bool Shader::Compiled() const
{
	return this->compiled;
}
inline GLuint Shader::GetProgram() const noexcept
{
	return this->program;
}

inline void Shader::SetActiveShader() const noexcept
{
	glUseProgram(this->program);
}

inline void Shader::SetInt(const std::string& name, const int i) const noexcept
{
	glUniform1i(this->UniformIndex(name), i);
}

inline void Shader::SetUnsignedInt(const std::string& name, const unsigned int i) const noexcept
{
	glUniform1ui(this->UniformIndex(name), i);
}

inline void Shader::SetFloat(const std::string& name, const float i) const noexcept
{
	glUniform1f(this->UniformIndex(name), i);
}

inline void Shader::SetVec2(const std::string& name, const glm::vec2& vec) const noexcept
{
	glUniform2fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
}

inline void Shader::SetVec3(const std::string& name, const glm::vec3& vec) const noexcept
{
	glUniform3fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
}

inline void Shader::SetVec4(const std::string& name, const glm::vec4& vec) const noexcept
{
	glUniform4fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
}

inline void Shader::SetMat4(const std::string& name, const glm::mat4& mat) const noexcept
{
	glUniformMatrix4fv(this->UniformIndex(name), 1, GL_FALSE, glm::value_ptr(mat));
}

inline void Shader::SetMat4s(const std::string& name, const std::span<glm::mat4> mats) const noexcept
{
	glUniformMatrix4fv(this->UniformIndex(name), static_cast<GLsizei>(mats.size()), GL_FALSE, glm::value_ptr(*mats.data()));
}

inline void Shader::SetTextureUnit(const std::string& name, const GLuint unit) const noexcept
{
	glUniform1i(this->UniformIndex(name), unit);
}

inline void Shader::SetTextureUnit(const std::string& name, const Texture2D& texture, const GLuint unit) const noexcept
{
	texture.BindTexture(unit);
	glUniform1i(this->UniformIndex(name), unit);
}

inline void Shader::SetTextureUnit(const std::string& name, const CubeMap& texture, const GLuint unit) const noexcept
{
	texture.BindTexture(unit);
	glUniform1i(this->UniformIndex(name), unit);
}

inline void Shader::UniformBlockBinding(const std::string& name, GLuint bindingPoint) const noexcept
{
	glUniformBlockBinding(this->program, this->UniformBlockIndex(name), bindingPoint);
}

inline void Shader::DrawArray(PrimitiveDrawingType type, const GLuint primitiveCount, const GLuint offset)
{
	glDrawArrays(static_cast<GLenum>(type), offset, primitiveCount);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArray(const GLuint primitiveCount, const GLuint offset)
{
	glDrawArrays(static_cast<GLenum>(type), offset, primitiveCount);
}


inline void Shader::DrawArray(PrimitiveDrawingType type, ArrayBuffer& buffer, const GLuint offset)
{
	this->DrawArray(type, buffer.GetElementCount(), offset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArray(ArrayBuffer& buffer, const GLuint elementOffset)
{
	this->DrawArray<type>(buffer.GetElementCount(), elementOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArrayInstanced(const GLuint primitiveCount, const GLuint instanceCount, 
																			const GLuint primitiveOffset, const GLuint instanceOffset)
{
	glDrawArraysInstancedBaseInstance(static_cast<GLenum>(type), primitiveOffset, primitiveCount, instanceCount, instanceOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArrayInstanced(ArrayBuffer& primitiveBuffer, ArrayBuffer& instanceBuffer,
																			const GLuint primitiveOffset, const GLuint instanceOffset)
{
	this->DrawArrayInstanced<type>(primitiveBuffer.GetElementCount(), instanceBuffer.GetElementCount(), primitiveOffset, instanceOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArrayIndirect(const IndirectDraw& parameters)
{
	glDrawArraysIndirect(static_cast<GLenum>(type), reinterpret_cast<const void*>(parameters));
}

inline void Shader::DrawElements(PrimitiveDrawingType type, ElementArray& indexBuffer, const GLuint elementOffset, const GLuint indexOffset)
{
	indexBuffer.BindBuffer();
	glDrawElementsBaseVertex(static_cast<GLenum>(type), indexBuffer.GetElementCount() - elementOffset, indexBuffer.GetElementType(),
		reinterpret_cast<void*>(indexBuffer.GetElementSize() * elementOffset), indexOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawElements(ElementArray& indexBuffer, const GLuint elementOffset, const GLuint indexOffset)
{
	indexBuffer.BindBuffer();
	glDrawElementsBaseVertex(static_cast<GLenum>(type), indexBuffer.GetElementCount() - elementOffset, indexBuffer.GetElementType(),
		reinterpret_cast<void*>(indexBuffer.GetElementSize() * elementOffset), indexOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawElementsInstanced(ElementArray& indexBuffer, ArrayBuffer& instanceBuffer,
	const GLuint elementOffset, const GLuint indexOffset, const GLint instanceOffset)
{
	indexBuffer.BindBuffer();
	glDrawElementsInstancedBaseVertexBaseInstance(static_cast<GLenum>(type),
		indexBuffer.GetElementCount() - elementOffset,
		indexBuffer.GetElementType(),
		reinterpret_cast<const void*>(indexBuffer.GetElementSize() * elementOffset),
		instanceBuffer.GetElementCount(),
		indexOffset,
		instanceOffset);
}

template<PrimitiveDrawingType type>
inline void Shader::DrawElements(const Buffer<BufferType::DrawIndirect>& indirect, const GLuint offset) const noexcept
{
	indirect.BindBuffer();
	glDrawElementsIndirect(static_cast<GLenum>(type), GL_UNSIGNED_INT, reinterpret_cast<const void*>(offset * sizeof(Elements)));
}

template<PrimitiveDrawingType type>
inline void Shader::MultiDrawElements(const Buffer<BufferType::DrawIndirect>& indirect) const noexcept
{
	this->MultiDrawElements(indirect, 0, indirect.GetElementCount());
}

template<PrimitiveDrawingType type>
inline void Shader::MultiDrawElements(const Buffer<BufferType::DrawIndirect>& indirect, const GLuint count) const noexcept
{
	this->MultiDrawElements(indirect, 0, count);
}

template<PrimitiveDrawingType type>
inline void Shader::MultiDrawElements(const Buffer<BufferType::DrawIndirect>& indirect, const GLuint offset, const GLuint count) const noexcept
{
	indirect.BindBuffer();
	glMultiDrawElementsIndirect(static_cast<GLenum>(type), GL_UNSIGNED_INT, reinterpret_cast<const void*>(offset * sizeof(Elements)),
		count, sizeof(Elements));
}

template<PrimitiveDrawingType type, class Container> inline void Shader::DrawElementsMemory(const Container& contents)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDrawElements(static_cast<GLenum>(type), static_cast<GLsizei>(contents.size()), GL_UNSIGNED_BYTE, reinterpret_cast<const void*>(contents.data()));
}

template<class Container> inline void Shader::DrawElementsMemory(PrimitiveDrawingType type, const Container& contents)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDrawElements(static_cast<GLenum>(type), static_cast<GLsizei>(contents.size()), GL_UNSIGNED_BYTE, contents.data());
}

#endif // FLAMES_SHADER_H
