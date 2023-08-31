#pragma once
#ifndef FLAMES_SHADER_H
#define FLAMES_SHADER_H
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <string>
#include "Texture2D.h"

// TODO: Fill out the rest of the enum
enum PrimitiveDrawingType : unsigned int
{
	Triangle          = GL_TRIANGLES,
	TriangleStrip     = GL_TRIANGLE_STRIP,
	TriangleAdjacency = GL_TRIANGLES_ADJACENCY,
	Lines             = GL_LINES,
	LineStrip         = GL_LINE_STRIP,
};

class Shader
{
protected:
	bool compiled, precompiled;
	GLuint program;
	std::string name;
	std::map<std::string, GLuint> mapping;
public:
	Shader();
	Shader(const std::string& name, bool forceRecompile = false);
	Shader(const std::string& vertex, const std::string& fragment, bool forceRecompile = false);
	Shader(const char* vertex, const char* fragment);
	Shader(const Shader& other) = delete;
	Shader(Shader&& other) noexcept;
	~Shader();

	Shader& operator=(const Shader& other) = delete;
	Shader& operator=(Shader&& other) noexcept;

	bool CompileSimple(const std::string& name, bool recompile = false);
	bool Compile(const std::string& vertex, const std::string& frag, bool recompile = false);
	bool CompileExplicit(const char* vertex, const char* fragment);

	constexpr bool Compiled() const;

	GLuint Index(const std::string& name) ;
	GLuint UniformIndex(const std::string& name) ;
	GLuint UniformBlockIndex(const std::string& name);

	void CalculateUniforms();
	void CleanUp();
	void ExportCompiled();

	inline GLuint GetProgram();
	inline void SetActiveShader();
	inline void SetInt(const std::string& name, const int i);
	inline void SetFloat(const std::string& name, const float i);
	inline void SetVec3(const std::string& name, const glm::vec3& vec);
	inline void SetMat4(const std::string& name, const glm::mat4& mat);
	inline void SetTextureUnit(const std::string& name, const GLuint unit);
	inline void SetTextureUnit(const std::string& name, Texture2D& texture, const GLuint unit);
	inline void UniformBlockBinding(const std::string& name, GLuint bindingPoint);

	static void IncludeInShaderFilesystem(const std::string& virtualName, const std::string& fileName);

	template<PrimitiveDrawingType type> inline void DrawElements(const GLuint num, const GLuint offset = 0);
	// TODO: Maybe concept?
	template<PrimitiveDrawingType type, class Container> inline void DrawElements(const Container& contents);
};

constexpr bool Shader::Compiled() const
{
	return this->compiled;
}
inline GLuint Shader::GetProgram()
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

inline void Shader::SetVec3(const std::string& name, const glm::vec3& vec)
{
	glUniform3fv(this->UniformIndex(name), 1, glm::value_ptr(vec));
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

template<PrimitiveDrawingType type>
inline void Shader::DrawElements(const GLuint num, const GLuint offset)
{
	glDrawArrays((GLenum)type, offset, num);
}

// TODO: Some kind of type inference thingy for index types bullshit
template<PrimitiveDrawingType type, class Container>
inline void Shader::DrawElements(const Container& contents)
{
	glDrawElements((GLenum) type, (GLsizei) contents.size(), GL_UNSIGNED_BYTE, contents.data());
}

#endif // FLAMES_SHADER_H
