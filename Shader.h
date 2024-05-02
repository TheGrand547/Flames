#pragma once
#ifndef FLAMES_SHADER_H
#define FLAMES_SHADER_H
#include <bit>
#include <filesystem>
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <string>
#include "Buffer.h"
#include "Texture2D.h"

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
	std::map<std::string, GLuint> mapping; // This is dumb

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

	// TODO: This sucks, don't do this

	// Compiles all stages available starting with the given filename
	// [name]v, [name]f,  [name]g,            [name]tc,             [name]te 
	// Vertex, fragment, geometry, tesselation control, tesselation evaluate, respectively
	bool CompileSimple(const std::string& name);


	bool Compile(const std::string& vertex, const std::string& frag);

	// TODO: Maybe later
	/*
	bool Compile(const std::string& vertex, const std::string& frag, const std::string& geometry);
	bool Compile(const std::string& vertex, const std::string& frag, const std::string& tessControl, const std::string& tessEval);
	bool Compile(const std::string& vertex, const std::string& frag, const std::string& geometry, const std::string& tessControl, const std::string& tessEval);
	*/
	// These all work on shader code as a string, not read in from a file
	bool CompileEmbedded(const std::string& vertex, const std::string& fragment);
	bool CompileEmbeddedGeometry(const std::string& vertex, const std::string& fragment, const std::string& geometry);
	bool CompileEmbeddedGeometryTesselation(const std::string& vertex, const std::string& fragment, const std::string& geometry, 
		const std::string& tessControl, const std::string& tessEval);
	bool CompileEmbeddedTesselation(const std::string& vertex, const std::string& fragment, const std::string& tessControl, const std::string& tessEval);

	constexpr bool Compiled() const;

	GLuint Index(const std::string& name);
	GLuint UniformIndex(const std::string& name);
	GLuint UniformBlockIndex(const std::string& name);

	void CalculateUniforms();
	void CleanUp();
	void ExportCompiled();

	inline GLuint GetProgram() const;
	inline void SetActiveShader();
	inline void SetInt(const std::string& name, const int i);
	inline void SetFloat(const std::string& name, const float i);
	inline void SetVec2(const std::string& name, const glm::vec2& vec);
	inline void SetVec3(const std::string& name, const glm::vec3& vec);
	inline void SetVec4(const std::string& name, const glm::vec4& vec);
	inline void SetMat4(const std::string& name, const glm::mat4& mat);
	inline void SetTextureUnit(const std::string& name, const GLuint unit);
	inline void SetTextureUnit(const std::string& name, Texture2D& texture, const GLuint unit);
	inline void UniformBlockBinding(const std::string& name, GLuint bindingPoint);

	// TODO: Maybe a draw function that takes a variable amount of VAO, Buffers, etc and does all that in a "magic" step

	// Feed the data in the buffer, unchanged, directly to the shader
	inline void DrawArray(PrimitiveDrawingType type, const GLuint primitiveCount, const GLuint elementOffset = 0);
	inline void DrawArray(PrimitiveDrawingType type, Buffer<ArrayBuffer>& buffer, const GLuint elementOffset = 0);
	template<PrimitiveDrawingType type> inline void DrawArray(const GLuint primitiveCount, const GLuint elementOffset = 0);
	template<PrimitiveDrawingType type> inline void DrawArray(Buffer<ArrayBuffer>& buffer, const GLuint elementOffset = 0);

	// Feed the shader the data in the buffer, augmented by the instance number, instance count times
	template<PrimitiveDrawingType type> inline void DrawArrayInstanced(const GLuint primitiveCount, const GLuint instanceCount, 
																		const GLuint primitiveOffset = 0, const GLuint instanceOffset = 0);
	template<PrimitiveDrawingType type> inline void DrawArrayInstanced(Buffer<ArrayBuffer>& primitiveBuffer, Buffer<ArrayBuffer>& instanceBuffer, 
																		const GLuint elementOffset = 0, const GLuint instanceOffset = 0);

	// Draw instanced data from the given parameter struct
	template<PrimitiveDrawingType type> inline void DrawArrayIndirect(const IndirectDraw& parameters);

	// TODO: maybe glDrawElementsRange() but probably not
	// TODO: Maybe concept?

	// Feed the data in the buffer to the shader in the order given by the passed container
	template<PrimitiveDrawingType type, class Container> inline void DrawElementsMemory(const Container& contents);
	template<class Container> inline void DrawElementsMemory(PrimitiveDrawingType type, const Container& contents);

	// Feed the data in the buffer to the shader in the order given by the ElementArray buffer
	template<PrimitiveDrawingType type> inline void DrawElements(Buffer<ElementArray>& indexBuffer, const GLuint elementOffset = 0, const GLuint indexOffset = 0);
	inline void DrawElements(PrimitiveDrawingType type, Buffer<ElementArray>& indexBuffer, const GLuint elementOffset = 0, const GLuint indexOffset = 0);

	template<PrimitiveDrawingType type> inline void DrawElementsInstanced(Buffer<ElementArray>& indexBuffer, Buffer<ArrayBuffer>& instanceBuffer,
															const GLuint elementOffset = 0, const GLuint indexOffset = 0, const GLint instanceOffset = 0);

	static void IncludeInShaderFilesystem(const std::string& virtualName, const std::string& fileName);
	static void SetBasePath(const std::string& basePath);
	static void SetRecompilationFlag(bool flag);
};

constexpr bool Shader::Compiled() const
{
	return this->compiled;
}
inline GLuint Shader::GetProgram() const
{
	return this->program;
}

inline void Shader::SetActiveShader()
{
	glUseProgram(this->program);
}

inline void Shader::SetInt(const std::string& name, const int i)
{
	glUniform1i(this->UniformIndex(name), i);
}

inline void Shader::SetFloat(const std::string& name, const float i)
{
	glUniform1f(this->UniformIndex(name), i);
}

inline void Shader::SetVec2(const std::string& name, const glm::vec2& vec)
{
	glUniform2fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
}

inline void Shader::SetVec3(const std::string& name, const glm::vec3& vec)
{
	glUniform3fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
}

inline void Shader::SetVec4(const std::string& name, const glm::vec4& vec)
{
	glUniform4fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
}

inline void Shader::SetMat4(const std::string& name, const glm::mat4& mat)
{
	glUniformMatrix4fv(this->UniformIndex(name), 1, GL_FALSE, glm::value_ptr(mat));
}


inline void Shader::SetTextureUnit(const std::string& name, const GLuint unit)
{
	glUniform1i(this->UniformIndex(name), unit);
}

inline void Shader::SetTextureUnit(const std::string& name, Texture2D& texture, const GLuint unit)
{
	texture.BindTexture(unit);
	glUniform1i(this->UniformIndex(name), unit);
}

inline void Shader::UniformBlockBinding(const std::string& name, GLuint bindingPoint)
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


inline void Shader::DrawArray(PrimitiveDrawingType type, Buffer<ArrayBuffer>& buffer, const GLuint offset)
{
	this->DrawArray(type, buffer.GetElementCount(), offset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArray(Buffer<ArrayBuffer>& buffer, const GLuint elementOffset)
{
	this->DrawArray<type>(buffer.GetElementCount(), elementOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArrayInstanced(const GLuint primitiveCount, const GLuint instanceCount, 
																			const GLuint primitiveOffset, const GLuint instanceOffset)
{
	glDrawArraysInstancedBaseInstance(static_cast<GLenum>(type), primitiveOffset, primitiveCount, instanceCount, instanceOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArrayInstanced(Buffer<ArrayBuffer>& primitiveBuffer, Buffer<ArrayBuffer>& instanceBuffer, 
																			const GLuint primitiveOffset, const GLuint instanceOffset)
{
	this->DrawArrayInstanced<type>(primitiveBuffer.GetElementCount(), instanceBuffer.GetElementCount(), primitiveOffset, instanceOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawArrayIndirect(const IndirectDraw& parameters)
{
	glDrawArraysIndirect(static_cast<GLenum>(type), reinterpret_cast<const void*>(parameters));
}

inline void Shader::DrawElements(PrimitiveDrawingType type, Buffer<ElementArray>& indexBuffer, const GLuint elementOffset, const GLuint indexOffset)
{
	indexBuffer.BindBuffer();
	glDrawElementsBaseVertex(static_cast<GLenum>(type), indexBuffer.GetElementCount() - elementOffset, indexBuffer.GetElementType(),
		reinterpret_cast<void*>(indexBuffer.GetElementSize() * elementOffset), indexOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawElements(Buffer<ElementArray>& indexBuffer, const GLuint elementOffset, const GLuint indexOffset)
{
	indexBuffer.BindBuffer();
	glDrawElementsBaseVertex(static_cast<GLenum>(type), indexBuffer.GetElementCount() - elementOffset, indexBuffer.GetElementType(),
		reinterpret_cast<void*>(indexBuffer.GetElementSize() * elementOffset), indexOffset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawElementsInstanced(Buffer<ElementArray>& indexBuffer, Buffer<ArrayBuffer>& instanceBuffer, 
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

// TODO: Some kind of type inference thingy for index types bullshit <- What does this mean
template<PrimitiveDrawingType type, class Container> inline void Shader::DrawElementsMemory(const Container& contents)
{
	//constexpr GLuint offset = 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	CheckError();
	glDrawElements(static_cast<GLenum>(type), static_cast<GLsizei>(contents.size()), GL_UNSIGNED_BYTE, reinterpret_cast<const void*>(contents.data()));
	CheckError();
}

template<class Container> inline void Shader::DrawElementsMemory(PrimitiveDrawingType type, const Container& contents)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDrawElements(static_cast<GLenum>(type), static_cast<GLsizei>(contents.size()), GL_UNSIGNED_BYTE, contents.data());
}

#endif // FLAMES_SHADER_H
