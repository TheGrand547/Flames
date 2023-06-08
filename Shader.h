#pragma once
#ifndef FLAMES_SHADER_H
#define FLAMES_SHADER_H
#include <glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <string>

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
	inline void SetActive();
	inline void SetInt(const std::string& name, const int i);
	inline void SetVec3(const std::string& name, const glm::vec3& vec);
	inline void SetMat4(const std::string& name, const glm::mat4& mat);
	inline void SetTextureUnit(const std::string& name, const GLuint unit);
	inline void UniformBlockBinding(const std::string& name, GLuint bindingPoint);
};

constexpr bool Shader::Compiled() const
{
	return this->compiled;
}
inline GLuint Shader::GetProgram()
{
	return this->program;
}

inline void Shader::SetActive()
{
	glUseProgram(this->program);
}

inline void Shader::SetInt(const std::string& name, const int i)
{
	glUniform1i(this->UniformIndex(name), i);
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

inline void Shader::UniformBlockBinding(const std::string& name, GLuint bindingPoint)
{
	glUniformBlockBinding(this->program, this->UniformBlockIndex(name), bindingPoint);
}

#endif // FLAMES_SHADER_H

