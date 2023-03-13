#pragma once
#ifndef FLAMES_SHADER_H
#define FLAMES_SHADER_H
#include <glew.h>
#include <string>
#include <glm/glm.hpp>

class Shader
{
protected:
	bool compiled, precompiled;
	GLuint program;
	std::string name;
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

	GLuint index(const std::string& name) const;
	GLuint uniformIndex(const std::string& name) const;

	void CleanUp();
	void SetActive();
	void ExportCompiled();


	void SetVec3(const std::string& name, const glm::vec3& vec) const;
	void SetMat4(const std::string& name, const glm::mat4& mat) const;
	void SetTextureUnit(const std::string& name, const unsigned int unit) const;
};

constexpr bool Shader::Compiled() const
{
	return this->compiled;
}

#endif // FLAMES_SHADER_H

