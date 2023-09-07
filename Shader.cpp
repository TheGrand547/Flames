#include "Shader.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <vector>
#include "log.h"

static std::map<std::string, std::string> shaderIncludeMapping;
static std::string shaderBasePath = "";

void Shader::SetBasePath(const std::string& basePath)
{
	shaderBasePath = basePath;
}


static void ApplyShaderIncludes(std::string& data)
{
	const std::string basis("#include \"{}\"\n");
	for (auto& [tag, mapping] : shaderIncludeMapping)
	{
		std::string current = std::format("#include \"{}\"\n", tag);
		std::size_t index = data.find(current);
		if (index != std::string::npos)
		{
			data.replace(index, current.length(), mapping);
		}
	}
	std::size_t index = data.find("#include \"");
	if (index != std::string::npos)
	{
		std::size_t end = data.find("\"", index);
		const std::size_t length = std::string("#include \"").length();
		std::filesystem::path path(data.substr(index + length, end - index - length));
		Shader::IncludeInShaderFilesystem(path.filename().string(), data.substr(index + length, end - index - length));
		std::string current = std::format("#include \"{}\"\n", path.filename().string());
		index = data.find(current);
		if (index != std::string::npos)
		{
			data.replace(index, current.length(), shaderIncludeMapping[path.filename().string()].c_str());
		}
	}
}

static GLuint CompileShader(GLenum type, std::string data)
{
	GLuint vertex = glCreateShader(type);
	// TODO: Test this
	ApplyShaderIncludes(data);

	const char* raw = data.c_str();

	glShaderSource(vertex, 1, &raw, nullptr);
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

Shader::Shader() : compiled(false), precompiled(false), program(0), mapping()
{

}
// TODO: Thingy to make sure that gaming happens

// If force is not set it will first check if a 'name.csp' (compiled shader program? no clue what industry standard is)
// Takes name for a shader, and reads the files 'namev.glsl' and 'namef.glsl'
Shader::Shader(const std::string& name, bool recompile) : compiled(false), precompiled(false), name(name), program(0), mapping()
{
	this->CompileSimple(name, recompile);
}

Shader::Shader(const std::string& vertex, const std::string& fragment, bool forceRecompile) : compiled(false), precompiled(false), 
																		name(""), program(0), mapping()
{
	this->Compile(vertex, fragment, forceRecompile);
}
Shader::Shader(const char* vertex, const char* fragment) : compiled(false), precompiled(false), name(""), program(0), mapping()
{
	this->CompileExplicit(vertex, fragment);
}

Shader::Shader(Shader&& other) noexcept : compiled(false), precompiled(false), name(""), program(0), mapping()
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
		this->mapping = other.mapping;
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
	std::filesystem::path compiledPath(shaderBasePath + combined + ".csp");
	std::filesystem::path vertexPath(shaderBasePath + vert + "v.glsl");
	std::filesystem::path fragmentPath(shaderBasePath + frag + "f.glsl");

	if (!(std::filesystem::exists(vertexPath) && std::filesystem::exists(fragmentPath)))
	{
		std::cerr << "One or more of the shader files missing for '" << name << "'" << std::endl;
		return false;
	}

	this->name = combined;

	std::ifstream input;
	if (!recompile && std::filesystem::exists(compiledPath)) // Attempt to read precompiled shader file
	{
		auto compiledTime = std::filesystem::last_write_time(compiledPath).time_since_epoch().count();
		auto vertexTime   = std::filesystem::last_write_time(vertexPath).time_since_epoch().count();
		auto fragmentTime = std::filesystem::last_write_time(fragmentPath).time_since_epoch().count();
		if (compiledTime > vertexTime && compiledTime > fragmentTime)
		{
			input.open(compiledPath.string(), std::ios::binary);
			if (input.is_open())
			{
				std::cout << "Reading '" << this->name << "' from compiled shader file." << std::endl;
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
	}

	std::ifstream vertexFile(vertexPath.string(), std::ifstream::in);
	std::ifstream fragmentFile(fragmentPath.string(), std::ifstream::in);
	if (vertexFile.is_open() && fragmentFile.is_open())
	{
		std::cout << "Compiling Shader from " << vertexPath << " and " << fragmentPath << std::endl;
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
			GLint logSize;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
			if (logSize)
			{
				GLchar* logMsg = new char[logSize];
				glGetProgramInfoLog(program, logSize, NULL, logMsg);
				std::cout << "Program Log: " << logMsg << std::endl;
				delete[] logMsg;
			}
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

// TODO: Maybe store in an unordered_map?
GLuint Shader::Index(const std::string& name) 
{
	return (GLuint) glGetAttribLocation(this->program, name.c_str());
}

GLuint Shader::UniformIndex(const std::string& name)
{
	if (this->mapping.find(name) != this->mapping.end())
	{
		return this->mapping[name];
	}
	this->mapping[name] = glGetUniformLocation(this->program, name.c_str());
	return this->mapping[name];
}

GLuint Shader::UniformBlockIndex(const std::string& name)
{
	return glGetUniformBlockIndex(this->program, name.c_str());
}

void Shader::CalculateUniforms()
{
	this->mapping.clear();
	const GLsizei bufferSize = 20;
	GLchar buffer[bufferSize];
	GLint count, size;
	GLsizei length;
	GLenum enumer;
	
	glGetProgramiv(this->program, GL_ACTIVE_UNIFORMS, &count);

	for (GLuint i = 0; i < (GLuint) count; i++)
	{
		glGetActiveUniform(this->program, i, bufferSize, &length, &size, &enumer, buffer);
		this->mapping[buffer] = i;
	}
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
	this->mapping.clear();
}

void Shader::ExportCompiled()
{
	if (!this->compiled || this->precompiled || !this->program || this->name == "")
		return;
	std::ofstream output(shaderBasePath + this->name + ".csp", std::ios::binary);
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
	else
	{
#ifndef RELEASE
		std::cout << "Failed to save Shader '" << this->name << "' to precompiled binary." << std::endl;
#endif
	}
	output.close();
}


// TODO: Use GL_ARB_shader_include or whatever if it's available idfk
void Shader::IncludeInShaderFilesystem(const std::string& virtualName, const std::string& fileName)
{
	if (shaderIncludeMapping.find(virtualName) != shaderIncludeMapping.end())
	{
		LogF("Already created a mapping with the name '%s'.\n", virtualName.c_str());
		return;
	}
	std::ifstream included(fileName, std::ifstream::in);
	if (included.is_open())
	{
		LogF("Including file '%s' in the virtual shader filesystem.\n", fileName.c_str());
		std::string text(std::istreambuf_iterator<char>{included}, {});
		shaderIncludeMapping[virtualName] = text;
	}
	else
	{
		LogF("Filename '%s' could not be found.\n", fileName.c_str());
	}
}
