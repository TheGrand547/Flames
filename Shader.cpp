#include "Shader.h"
#include <filesystem>
#include <fstream>
#include <GL.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

static GLuint CompileShader(GLenum type, const char* data)
{
	GLuint vertex = glCreateShader(type);

	glShaderSource(vertex, 1, &data, nullptr);
	glCompileShader(vertex);
	int success;

	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		GLint length;
		glGetShaderiv(vertex, GL_INFO_LOG_LENGTH, &length);
		char* infoLog = new char[length + 1];
		infoLog[length] = '\0';
		glGetShaderInfoLog(vertex, length, NULL, infoLog);
		std::cout << "Compilation of Shader failed\n" << infoLog << std::endl;
		delete[] infoLog;
		return 0; // Error Code
	}
	return vertex;
}

Shader::Shader() : compiled(false), precompiled(false), program(0)
{

}

// If force is not set it will first check if a 'name.csp' (compiled shader program? no clue what industry standard is)
// Takes name for a shader, and reads the files 'namev.glsl' and 'namef.glsl'
Shader::Shader(const std::string& name, bool recompile) : compiled(false), precompiled(false), name(name), program(0)
{
	this->CompileSimple(name, recompile);
}

Shader::Shader(const std::string& vertex, const std::string& fragment, bool forceRecompile) : compiled(false), precompiled(false), name(""), program(0)
{
	this->Compile(vertex, fragment, forceRecompile);
}
Shader::Shader(const char* vertex, const char* fragment) : compiled(false), precompiled(false), name(""), program(0)
{
	this->CompileExplicit(vertex, fragment);
}

Shader::Shader(Shader&& other) noexcept : compiled(false), precompiled(false), name(""), program(0)
{
	*this = std::forward<Shader>(other);
}

Shader::~Shader()
{
	this->CleanUp();
}

Shader& Shader::operator=(Shader&& other) noexcept
{
	if (this != &other)
	{
		this->CleanUp();
		this->program = other.program;
		this->compiled = other.compiled;
		this->precompiled = other.precompiled;
		this->name = other.name;
		other.program = 0;
		other.CleanUp();
	}
	return *this;
}

bool Shader::CompileSimple(const std::string& name, bool recompile)
{
	return this->Compile(name, name, recompile);
}

bool Shader::Compile(const std::string& vert, const std::string& frag, bool recompile)
{
	this->CleanUp();
	std::string combined = (vert == frag) ? vert : vert + frag;
	std::filesystem::path compiledPath(combined + ".csp");
	std::filesystem::path vertexPath(vert + "v.glsl");
	std::filesystem::path fragmentPath(frag + "f.glsl");

	if (!(std::filesystem::exists(vertexPath) && std::filesystem::exists(fragmentPath)))
	{
		std::cerr << "One or more of the shader files missing for '" << name << "'" << std::endl;
		return false;
	}

	this->name = combined;

	std::ifstream input;
	if (!recompile && (std::filesystem::exists(compiledPath) && \
		(std::filesystem::last_write_time(compiledPath) > std::filesystem::last_write_time(fragmentPath)
			|| std::filesystem::last_write_time(compiledPath) > std::filesystem::last_write_time(vertexPath)))) // Attempt to read precompiled shader file
	{
		input.open(compiledPath.string(), std::ios::binary);
		if (input.is_open())
		{
			GLint length = 0;
			GLenum format = 0;
			input.read((char*)&length, sizeof(GLint));
			input.read((char*)&format, sizeof(GLenum));
			char* data = new char[length];
			input.read(data, length);
			this->program = glCreateProgram();
			glProgramBinary(this->program, format, reinterpret_cast<void*>(data), length);
			delete[] data;
			input.close();

			int result;
			glGetProgramiv(this->program, GL_LINK_STATUS, &result);
			if (result)
			{
				this->compiled = true;
				this->precompiled = true;
				return true;
			}
			GLint logSize;
			glGetProgramiv(this->program, GL_INFO_LOG_LENGTH, &logSize);
			char* logMsg = new char[logSize];
			glGetProgramInfoLog(this->program, logSize, NULL, logMsg);
			std::cerr << "Error reading compiled shader from file '" << name << ".csp'" << std::endl << logMsg << std::endl;
			delete[] logMsg;
			input.close();
			this->program = 0;
		}
	}

	std::ifstream vertexFile(vertexPath.string(), std::ifstream::in);
	std::ifstream fragmentFile(fragmentPath.string(), std::ifstream::in);
	if (vertexFile.is_open() && fragmentFile.is_open())
	{
		std::string vertex(std::istreambuf_iterator<char>{vertexFile}, {});
		std::string fragment(std::istreambuf_iterator<char>{fragmentFile}, {});
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex.c_str());
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment.c_str());
		if (vShader && fShader && (this->program = glCreateProgram()))
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glLinkProgram(this->program);
			int result;
			glGetProgramiv(this->program, GL_LINK_STATUS, &result);
			if (!result)
			{
				GLint logSize;
				glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
				char* logMsg = new char[logSize];
				glGetProgramInfoLog(program, logSize, NULL, logMsg);
				std::cerr << logMsg << std::endl;
				delete[] logMsg;
				this->program = 0;
				return false;
			}
			glDeleteShader(vShader);
			glDeleteShader(fShader);
			this->compiled = true;
			this->ExportCompiled();
		}
		vertexFile.close();
		fragmentFile.close();
	}
	return this->compiled;
}

bool Shader::CompileExplicit(const char* vertex, const char* fragment)
{
	this->precompiled = false;
	GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex);
	GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment);
	if (vShader && fShader && (this->program = glCreateProgram()))
	{
		glAttachShader(this->program, vShader);
		glAttachShader(this->program, fShader);
		glLinkProgram(this->program);
		int result;
		glGetProgramiv(this->program, GL_LINK_STATUS, &result);
		if (!result)
		{
			GLint logSize;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
			char* logMsg = new char[logSize];
			glGetProgramInfoLog(program, logSize, NULL, logMsg);
			std::cerr << logMsg << std::endl;
			delete[] logMsg;
			this->program = 0;
			return false;
		}
		glDeleteShader(vShader);
		glDeleteShader(fShader);
		this->compiled = true;
	}
	return this->compiled;
}


GLuint Shader::index(const std::string& name) const
{
	return glGetAttribLocation(this->program, name.c_str());
}

GLuint Shader::uniformIndex(const std::string& name) const
{
	return glGetUniformLocation(this->program, name.c_str());
}

void Shader::CleanUp()
{
	if (this->program)
	{
		glDeleteProgram(this->program);
		this->program = 0;
		this->compiled = false;
		this->precompiled = false;
		this->name = "";
	}
}

void Shader::SetActive()
{
	glUseProgram(this->program);
}

void Shader::ExportCompiled()
{
	if (!this->compiled || this->precompiled || !this->program || this->name == "")
		return;
	std::ofstream output(this->name + ".csp", std::ios::binary);
	if (output.is_open())
	{
		GLint length = 0;
		GLenum format;
		glGetProgramiv(this->program, GL_PROGRAM_BINARY_LENGTH, &length);
		std::vector<char> buffer(length);
		glGetProgramBinary(this->program, length, &length, &format, buffer.data());
		output.write(reinterpret_cast<char*> (&length), sizeof(length));
		output.write(reinterpret_cast<char*> (&format), sizeof(format));
		output.write(buffer.data(), buffer.size());
	}
	output.close();
}

void Shader::SetVec3(const std::string& name, const glm::vec3& vec) const
{
	glUniform3fv(this->uniformIndex(name), 1, glm::value_ptr(vec));
}

void Shader::SetMat4(const std::string& name, const glm::mat4& mat) const
{
	glUniformMatrix4fv(this->uniformIndex(name), 1, GL_FALSE, glm::value_ptr(mat));
}

void Shader::SetTextureUnit(const std::string& name, const unsigned int unit) const
{
	glUniform1ui(this->uniformIndex(name), unit);
}
