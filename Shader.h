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

enum PrimitiveDrawingType : unsigned int
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

// Compute shaders will be their own thing

enum ShaderStages : unsigned char
{
	None                   = 0,
	Geometry               = 1,
	Tesselation            = 2,
	GeometryAndTesselation = Geometry | Tesselation,
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

	bool Compile(const std::string& vertex, const std::string& frag, const std::string& geometry);
	bool Compile(const std::string& vertex, const std::string& frag, const std::string& tessControl, const std::string& tessEval);
	bool Compile(const std::string& vertex, const std::string& frag, const std::string& geometry, const std::string& tessControl, const std::string& tessEval);
	
	// These all work on shader code as a string, not read in from a file
	bool CompileEmbedded(const std::string& vertex, const std::string& fragment);
	bool CompileEmbeddedGeometry(const std::string& vertex, const std::string& fragment, const std::string& geometry);
	bool CompileEmbeddedGeometryTesselation(const std::string& vertex, const std::string& fragment, const std::string& geometry, 
		const std::string& tessControl, const std::string& tessEval);
	bool CompileEmbeddedTesselation(const std::string& vertex, const std::string& fragment, const std::string& tessControl, const std::string& tessEval);

	constexpr bool Compiled() const;

	GLuint Index(const std::string& name) ;
	GLuint UniformIndex(const std::string& name) ;
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

	inline void DrawElements(PrimitiveDrawingType type, const GLuint num, const GLuint elementOffset = 0);
	inline void DrawElements(PrimitiveDrawingType type, Buffer<ArrayBuffer>& buffer, const GLuint elementOffset = 0);

	template<PrimitiveDrawingType type> inline void DrawElements(const GLuint num, const GLuint elementOffset = 0);
	template<PrimitiveDrawingType type> inline void DrawElements(Buffer<ArrayBuffer>& buffer, const GLuint elementOffset = 0);

	// TODO: maybe glDrawElementsRange() but probably not
	// TODO: Maybe concept?
	template<PrimitiveDrawingType type, class Container> inline void DrawIndexedMemory(const Container& contents);
	template<class Container> inline void DrawIndexedMemory(PrimitiveDrawingType type, const Container& contents);

	template<PrimitiveDrawingType type> inline void DrawIndexed(Buffer<ElementArray>& buffer, const GLuint elementOffset = 0);
	inline void DrawIndexed(PrimitiveDrawingType type, Buffer<ElementArray>& buffer, const GLuint elementOffset = 0);

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

inline void Shader::DrawElements(PrimitiveDrawingType type, const GLuint num, const GLuint offset)
{
	glDrawArrays((GLenum)type, offset, num);
}

template<PrimitiveDrawingType type> inline void Shader::DrawElements(const GLuint num, const GLuint offset)
{
	glDrawArrays((GLenum) type, offset, num);
}


inline void Shader::DrawElements(PrimitiveDrawingType type, Buffer<ArrayBuffer>& buffer, const GLuint offset)
{
	this->DrawElements(type, buffer.GetElementCount(), offset);
}

template<PrimitiveDrawingType type> inline void Shader::DrawElements(Buffer<ArrayBuffer>& buffer, const GLuint elementOffset)
{
	this->DrawElements<type>(buffer.GetElementCount(), elementOffset);
}

inline void Shader::DrawIndexed(PrimitiveDrawingType type, Buffer<ElementArray>& buffer, const GLuint elementOffset)
{
	buffer.BindBuffer();
	glDrawElements((GLenum) type, buffer.GetElementCount() - elementOffset, buffer.GetElementType(), 
			(const void*) (buffer.GetElementSize() * elementOffset));
}

template<PrimitiveDrawingType type> inline void Shader::DrawIndexed(Buffer<ElementArray>& buffer, const GLuint elementOffset)
{
	buffer.BindBuffer();
	glDrawElements((GLenum)type, buffer.GetElementCount() - elementOffset, buffer.GetElementType(), 
		(const void*)(buffer.GetElementSize() * elementOffset));
}

// TODO: Some kind of type inference thingy for index types bullshit <- What does this mean
template<PrimitiveDrawingType type, class Container> inline void Shader::DrawIndexedMemory(const Container& contents)
{
	//constexpr GLuint offset = 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDrawElements((GLenum) type, (GLsizei) contents.size(), GL_UNSIGNED_BYTE, contents.data());
}

template<class Container> inline void Shader::DrawIndexedMemory(PrimitiveDrawingType type, const Container& contents)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDrawElements((GLenum)type, (GLsizei)contents.size(), GL_UNSIGNED_BYTE, contents.data());
}

#endif // FLAMES_SHADER_H
