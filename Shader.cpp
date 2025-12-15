#include "Shader.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <vector>
#include "log.h"

#ifdef RELEASE
#define EXIT
#else // RELEASE
//#define EXIT exit(-1)
#define EXIT {system("pause"); exit(-1);}
#endif // RELEASE

static std::map<std::string, std::string> shaderIncludeMapping;
static std::string shaderBasePath = "";
static bool Recompile = false;
static const std::array<std::string, 5> extensions = { "v.glsl", "f.glsl", "g.glsl", "tc.glsl", "te.glsl" };
static constexpr std::array<GLenum, 5> shaderType = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_GEOMETRY_SHADER,
	GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER };

static std::vector<std::vector<std::string>> shaderDefines = { std::vector<std::string>{} };
static std::vector<std::string> shaderTempDefines;

void Shader::SetBasePath(const std::string& basePath)
{
	shaderBasePath = basePath + FILEPATH_SLASH;
}

void Shader::ForceRecompile(bool flag)
{
	Recompile = flag;
}

void Shader::Define(const std::string& define)
{
	shaderDefines.back().push_back(define);
}

void Shader::DefineTemp(const std::string& define)
{
	shaderTempDefines.push_back(define);
}

void Shader::PushContext()
{
	shaderDefines.push_back({});
}

void Shader::PopContext()
{
	if (shaderDefines.size() > 0)
	{
		shaderDefines.pop_back();
	}
}

static void ApplyShaderIncludes(std::string& data)
{
	if (shaderDefines.size() != 0 || shaderTempDefines.size() != 0)
	{
		std::stringstream newDefines;
		newDefines << '\n';
		for (const auto& defineContext : shaderDefines)
		{
			for (const auto& define : defineContext)
			{
				newDefines << define << '\n';
			}
		}
		for (const auto& define : shaderTempDefines)
		{
			newDefines << define << '\n';
		}
		shaderTempDefines.clear();
		newDefines << '\n';
		std::size_t first = data.find('\n');
		if (first == std::string::npos)
		{
			Log("No newlines in the given shader.");
			EXIT;
		}
		else
		{
			data.insert(first, newDefines.str());
		}
	}
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
		std::string current = std::format("#include \"{}\"", path.filename().string());
		index = data.find(current);
		if (index != std::string::npos)
		{
			data.replace(index, current.length(), shaderIncludeMapping[path.filename().string()].c_str());
		}
	}
}

static constexpr std::string_view FailedShaderLog = "failed_shader.txt";

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

		std::unique_ptr<char[]> infoLog = std::make_unique<char[]>(static_cast<size_t>(length) + 1);
		infoLog[length] = '\0';
		glGetShaderInfoLog(vertex, length, NULL, infoLog.get());
		std::cout << "Compilation of Shader failed\n" << std::string(infoLog.get()) << '\n';

		std::ofstream grumpiest(FailedShaderLog.data());
		if (grumpiest.is_open())
		{
			std::cout << "See error file '" << FailedShaderLog << "' for the full shader source being compiled.\n";
			grumpiest << data;
			grumpiest.close();
		}
		else
		{
			std::cout << data << '\n';
		}
		EXIT;
		return 0; // Error Code
	}
	return vertex;
}

Shader::Shader(ShaderStages stages) : compiled(false), precompiled(false), program(0), stages(stages)
{

}

// If force is not set it will first check if a 'name.csp' (compiled shader program? no clue what industry standard is)
// Takes name for a shader, and reads the files 'namev.glsl' and 'namef.glsl'
Shader::Shader(const std::string& name) : compiled(false), precompiled(false), name(name), program(0)
{
	this->CompileSimple(name);
}

Shader::Shader(const std::string& vertex, const std::string& fragment) : compiled(false), precompiled(false), 
																		name(""), program(0)
{
	this->Compile(vertex, fragment);
}
Shader::Shader(const char* vertex, const char* fragment) : compiled(false), precompiled(false), name(""), program(0)
{
	this->CompileEmbedded(vertex, fragment);
}

Shader::Shader(Shader&& other) noexcept : compiled(false), precompiled(false), name(""), program(0)
{
	*this = std::forward<Shader>(other);
}

Shader::~Shader()
{
	// TODO: Return to this
	//this->ExportCompiled();
	//Log("{} is dyin", this->name);
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

bool Shader::TryLoadCompiled(const std::string& name, std::chrono::system_clock::rep threshold)
{
	std::filesystem::path compiledPath(shaderBasePath + name + ".csp");
	if (!Recompile && std::filesystem::exists(compiledPath)) // Attempt to read precompiled shader file
	{
		std::chrono::system_clock::rep compiledTime = std::filesystem::last_write_time(compiledPath).time_since_epoch().count();
		if (compiledTime > threshold)
		{
			std::ifstream input;
			input.open(compiledPath, std::ios::binary);
			if (input.is_open())
			{
				Log("Loaded shader from file '{}'", compiledPath.string());
				GLint length = 0;
				GLenum format = 0;
				input.read(reinterpret_cast<char*>(&length), sizeof(GLint));
				input.read(reinterpret_cast<char*>(&format), sizeof(GLenum));

				std::unique_ptr<char[]> data = std::make_unique<char[]>(static_cast<size_t>(length) + 1);
				data[length] = '\0';
				input.read(data.get(), length);
				this->program = glCreateProgram();
				glProgramBinary(this->program, format, reinterpret_cast<void*>(data.get()), length);
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
				std::unique_ptr<char[]> logMsg = std::make_unique<char[]>(static_cast<size_t>(logSize) + 1);
				logMsg[length] = '\0';
				glGetProgramInfoLog(this->program, logSize, NULL, logMsg.get());
				Log("Error reading compiled shader from file '{}.csp' \n{}", name, logMsg.get());
				input.close();
				this->program = 0;
			}
		}
	}
	return false;
}

bool Shader::ProgramStatus()
{
	int result = 0;
	glGetProgramiv(this->program, GL_LINK_STATUS, &result);
	if (!result)
	{
		GLint logSize;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
		std::size_t trueLogSize = static_cast<std::size_t>(logSize) + 1;
		std::unique_ptr<char[]> logMsg = std::make_unique<char[]>(trueLogSize);
		logMsg[trueLogSize - 1] = '\0';
		glGetProgramInfoLog(program, logSize, NULL, logMsg.get());
		Log("Linking of shader failed: \n{}", logMsg.get());
		this->program = 0;
		//EXIT;
	}
	GLint logSize = 0;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
	if (logSize)
	{
		std::size_t trueLogSize = static_cast<std::size_t>(logSize) + 1;
		std::unique_ptr<GLchar[]> logMsg = std::make_unique<GLchar[]>(trueLogSize);
		logMsg[trueLogSize - 1] = '\n';
		glGetProgramInfoLog(program, logSize, NULL, logMsg.get());
		Log("Program Log: \n{}", logMsg.get());
		this->program = 0;
		//EXIT;
	}
	this->compiled = result;
	return result;
}

bool Shader::CompileSimple(const std::string& name)
{
	this->CleanUp();
	this->name = name;
#ifdef RELEASE
	#pragma message("Compile Simple should not be used in release mode! Please embed the shaders.")
#endif // RELEASE
	Log("Compiling Shader '{}'", name);
	{
		std::chrono::system_clock::rep timer = 0;
		int mask = 0;
		for (std::size_t i = 0; i < 5; i++)
		{
			std::filesystem::path localPath(shaderBasePath + name + extensions[i]);
			if (std::filesystem::exists(localPath))
			{
				timer = std::max(timer, std::filesystem::last_write_time(localPath).time_since_epoch().count());
				mask |= (1 << i);
			}
		}
		// Must have fragment and vertex shaders present
		if (!this->TryLoadCompiled(name, timer) && (mask & 3) == 3)
		{
			// These *must* exist
			std::filesystem::path vertexPath(shaderBasePath + name + extensions[0]);
			std::filesystem::path fragmentPath(shaderBasePath + name + extensions[1]);
			std::ifstream vertexFile(vertexPath.string(), std::ifstream::in);
			std::ifstream fragmentFile(fragmentPath.string(), std::ifstream::in);
			std::string vertex(std::istreambuf_iterator<char>{vertexFile}, {});
			std::string fragment(std::istreambuf_iterator<char>{fragmentFile}, {});
			vertexFile.close();
			fragmentFile.close();
			switch ((mask >> 2))
			{
				// Only Geometry is present
			case 1:
			{
				std::filesystem::path geometryPath(shaderBasePath + name + extensions[2]);
				std::ifstream geometryFile(geometryPath.string(), std::ifstream::in);
				std::string geometry(std::istreambuf_iterator<char>{geometryFile}, {});
				this->CompileEmbeddedGeometry(vertex, fragment, geometry);
				geometryFile.close();
				break;
			}
			// Tess control and evaluation are present
			case 6:
			{
				std::filesystem::path tcPath(shaderBasePath + name + extensions[3]);
				std::filesystem::path tePath(shaderBasePath + name + extensions[4]);
				std::ifstream teFile(tePath.string(), std::ifstream::in);
				std::ifstream tcFile(tcPath.string(), std::ifstream::in);
				std::string te(std::istreambuf_iterator<char>{teFile}, {});
				std::string tc(std::istreambuf_iterator<char>{tcFile}, {});
				this->CompileEmbeddedTesselation(vertex, fragment, tc, te);
				teFile.close();
				tcFile.close();
				break;
			}
			// All are present
			case 7:
			{
				std::filesystem::path geometryPath(shaderBasePath + name + extensions[2]);
				std::filesystem::path tcPath(shaderBasePath + name + extensions[3]);
				std::filesystem::path tePath(shaderBasePath + name + extensions[4]);
				std::ifstream geometryFile(geometryPath.string(), std::ifstream::in);
				std::ifstream teFile(tePath.string(), std::ifstream::in);
				std::ifstream tcFile(tcPath.string(), std::ifstream::in);
				std::string geometry(std::istreambuf_iterator<char>{geometryFile}, {});
				std::string te(std::istreambuf_iterator<char>{teFile}, {});
				std::string tc(std::istreambuf_iterator<char>{tcFile}, {});
				this->CompileEmbeddedGeometryTesselation(vertex, fragment, geometry, tc, te);
				geometryFile.close();
				teFile.close();
				tcFile.close();
				break;
			}
			// No extra stages, or invalid combination thereof, in which case default to just vertex/fragment
			default:
				if ((mask >> 2) != 0)
				{
					Log("Missing one of the tesselation shader stages for '{}'", name);
				}
				this->CompileEmbedded(vertex, fragment);
				break;
			}

		}
		else
		{
			Log("Something went wrong compiling '{}'", this->name);
			EXIT;
		}
		//return false;
	}
	if (this->compiled)
	{
		this->ExportCompiled();
	}
	return this->compiled;
}

bool Shader::CompileCompute(const std::string& name)
{
	std::filesystem::path path(shaderBasePath + name + ".glsl");
	if (!std::filesystem::exists(path))
	{
		Log("Failed to load shader '{}'", name);
		return false;
	}
	std::ifstream inputs(path.string(), std::ifstream::in);
	if (inputs.is_open())
	{
		Log("Compiling Shader '{}'", name);
		std::string compute(std::istreambuf_iterator<char>{inputs}, {});
		this->name = name;
		inputs.close();
		return this->CompileComputeEmbedded(compute);
	}
	return false;
}

bool Shader::CompileComputeEmbedded(const std::string& source)
{
	std::string local(source);
	ApplyShaderIncludes(local);
	GLuint compute = CompileShader(GL_COMPUTE_SHADER, local.c_str());
	if (compute && (this->program = glCreateProgram()))
	{
		glAttachShader(this->program, compute);
		glLinkProgram(this->program);
		this->ProgramStatus();
		glDeleteShader(compute);
	}
	return this->compiled;
}

void Shader::DispatchCompute(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept
{
	this->SetActiveShader();
	glDispatchCompute(x, y, z);
}

bool Shader::Compile(const std::string& vert, const std::string& frag, bool instanced)
{
	this->CleanUp();
	std::string combined = (vert == frag) ? vert : vert + frag;
	if (instanced)
	{
		combined += "_instanced";
		Shader::DefineTemp("#define INSTANCED");
	}
	std::filesystem::path compiledPath(shaderBasePath + combined + ".csp");
	std::filesystem::path vertexPath(shaderBasePath + vert + extensions[0]);
	std::filesystem::path fragmentPath(shaderBasePath + frag + extensions[1]);

	if (!(std::filesystem::exists(vertexPath) && std::filesystem::exists(fragmentPath)))
	{
		Log("One or more of the shader files missing for '{}'", combined);
		EXIT;
		return false;
	}

	this->name = combined;

	auto newestWrite = std::max(std::filesystem::last_write_time(vertexPath).time_since_epoch().count(),
		std::filesystem::last_write_time(fragmentPath).time_since_epoch().count());

	if (TryLoadCompiled(combined, newestWrite))
	{
		return true;
	}
	
	std::ifstream vertexFile(vertexPath.string(), std::ifstream::in);
	std::ifstream fragmentFile(fragmentPath.string(), std::ifstream::in);
	if (vertexFile.is_open() && fragmentFile.is_open())
	{
		Log("Compiling Shader from {} and {}", vertexPath.string(), fragmentPath.string());
		std::string vertex(std::istreambuf_iterator<char>{vertexFile}, {});
		std::string fragment(std::istreambuf_iterator<char>{fragmentFile}, {});
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex.c_str());
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment.c_str());
		if (vShader && fShader && (this->program = glCreateProgram()))
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glLinkProgram(this->program);
			this->ProgramStatus();
			glDeleteShader(vShader);
			glDeleteShader(fShader);
			this->ExportCompiled();
		}
		vertexFile.close();
		fragmentFile.close();
	}
	return this->compiled;
}

bool Shader::CompileEmbedded(const std::string& vertex, const std::string& fragment)
{
	if ((this->program = glCreateProgram()))
	{
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex);
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment);
		if (vShader && fShader)
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glLinkProgram(this->program);
			this->ProgramStatus();
			glDeleteShader(vShader);
			glDeleteShader(fShader);
		}
	}
	return this->compiled;
}

bool Shader::CompileEmbeddedGeometry(const std::string& vertex, const std::string& fragment, const std::string& geometry)
{
	this->precompiled = false;
	if ((this->program = glCreateProgram()))
	{
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex);
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment);
		GLuint gShader = CompileShader(GL_GEOMETRY_SHADER, geometry);
		if (vShader && fShader && gShader)
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glAttachShader(this->program, gShader);
			glLinkProgram(this->program);
			this->ProgramStatus();
			glDeleteShader(vShader);
			glDeleteShader(fShader);
			glDeleteShader(gShader);
		}
	}
	return this->compiled;
}

bool Shader::CompileEmbeddedGeometryTesselation(const std::string& vertex, const std::string& fragment, const std::string& geometry, 
	const std::string& tessControl, const std::string& tessEval)
{
	this->precompiled = false;
	if ((this->program = glCreateProgram()))
	{
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex);
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment);
		GLuint gShader = CompileShader(GL_FRAGMENT_SHADER, geometry);
		GLuint tcShader = CompileShader(GL_TESS_CONTROL_SHADER, tessControl);
		GLuint teShader = CompileShader(GL_TESS_EVALUATION_SHADER, tessEval);
		if (vShader && fShader && gShader && tcShader && teShader)
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glAttachShader(this->program, gShader);
			glAttachShader(this->program, tcShader);
			glAttachShader(this->program, teShader);
			glLinkProgram(this->program);
			this->ProgramStatus();
			glDeleteShader(vShader);
			glDeleteShader(fShader);
			glDeleteShader(gShader);
			glDeleteShader(teShader);
			glDeleteShader(tcShader);
		}
	}
	return this->compiled;
}

bool Shader::CompileEmbeddedTesselation(const std::string& vertex, const std::string& fragment, const std::string& tessControl, const std::string& tessEval)
{
	this->precompiled = false;
	if ((this->program = glCreateProgram()))
	{
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, vertex);
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, fragment);
		GLuint tcShader = CompileShader(GL_TESS_CONTROL_SHADER, tessControl);
		GLuint teShader = CompileShader(GL_TESS_EVALUATION_SHADER, tessEval);
		if (vShader && fShader && tcShader && teShader)
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glAttachShader(this->program, tcShader);
			glAttachShader(this->program, teShader);
			glLinkProgram(this->program);
			this->ProgramStatus();
			glDeleteShader(vShader);
			glDeleteShader(fShader);
			glDeleteShader(teShader);
			glDeleteShader(tcShader);
		}
	}
	return this->compiled;
}

bool Shader::CompileSingleFile(const std::string& filename, bool instanced)
{
	this->CleanUp();
	std::filesystem::path inputPath(shaderBasePath + filename + ".glsl");

	if (!std::filesystem::exists(inputPath))
	{
		Log("File is missing '{}'", inputPath.string());
		EXIT;
		return false;
	}
	std::string compiled_name = filename;
	if (instanced)
	{
		compiled_name += "_instanced";
	}
	std::filesystem::path compiledPath(shaderBasePath + compiled_name + ".csp");

	this->name = compiled_name;

	auto newestWrite = std::filesystem::last_write_time(inputPath).time_since_epoch().count();

	// TODO: Need to have some kind of flag thing that separates the precompiled versions from each other
	if (TryLoadCompiled(this->name, newestWrite))
	{
		return true;
	}

	std::ifstream inputFile(inputPath.string(), std::ifstream::in);
	if (inputFile.is_open())
	{
		Log("Compiling Shader from '{}'", filename);
		std::string inputA(std::istreambuf_iterator<char>{inputFile}, {});
		if (instanced)
		{
			Shader::DefineTemp("#define INSTANCED");
		}
		ApplyShaderIncludes(inputA);

		Shader::DefineTemp("#define VERTEX");
		GLuint vShader = CompileShader(GL_VERTEX_SHADER, inputA.c_str());
		Shader::DefineTemp("#define FRAGMENT");
		GLuint fShader = CompileShader(GL_FRAGMENT_SHADER, inputA.c_str());
		if (vShader && fShader && (this->program = glCreateProgram()))
		{
			glAttachShader(this->program, vShader);
			glAttachShader(this->program, fShader);
			glLinkProgram(this->program);
			if (!this->ProgramStatus())
			{
				std::string outrun = inputA.c_str();
				ApplyShaderIncludes(outrun);
				Log("\n\n{}\n\n", outrun);
				EXIT;
			}
			glDeleteShader(vShader);
			glDeleteShader(fShader);
			this->ExportCompiled();
		}
		inputFile.close();
	}
	else
	{
		Log("Some kind of error opening file '{}'", inputPath.string());
	}
	return this->compiled;
}


GLuint Shader::Index(const std::string& name)  const
{
	return static_cast<GLuint>(glGetAttribLocation(this->program, name.c_str()));
}

GLuint Shader::UniformIndex(const std::string& name) const
{
	return glGetUniformLocation(this->program, name.c_str());
}

GLuint Shader::UniformBlockIndex(const std::string& name) const
{
	return glGetUniformBlockIndex(this->program, name.c_str());
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

void Shader::ExportCompiled() const
{
	if (!this->compiled || this->precompiled || !this->program || this->name == "")
		return;
	Log("Exporting Shader '{}'", this->name);
	std::ofstream output(shaderBasePath + this->name + ".csp", std::ios::binary);
	if (output.is_open())
	{
		GLint length = 0;
		GLenum format;
		glGetProgramiv(this->program, GL_PROGRAM_BINARY_LENGTH, &length);
		std::size_t trueLength = static_cast<std::size_t>(length) + 1;
		std::unique_ptr<char[]> programData = std::make_unique<char[]>(trueLength);
		glGetProgramBinary(this->program, length, &length, &format, programData.get());
		output.write(reinterpret_cast<char*> (&length), sizeof(length));
		output.write(reinterpret_cast<char*> (&format), sizeof(format));
		output.write(programData.get(), trueLength);
	}
	else
	{
#ifndef RELEASE
		Log("Failed to save Shader '{}' to precompiled binary\n", this->name);
#endif
	}
	output.close();
}


// TODO: Use GL_ARB_shader_include or whatever if it's available idfk
void Shader::IncludeInShaderFilesystem(const std::string& virtualName, const std::string& fileName)
{
	if (shaderIncludeMapping.find(virtualName) != shaderIncludeMapping.end())
	{
		Log("Already created a mapping with the name '{}'.", virtualName);
		return;
	}
	std::ifstream included(shaderBasePath + fileName, std::ifstream::in);
	if (included.is_open())
	{
		//Log("Including file '{}' in the virtual shader filesystem.\n", fileName);
		std::string text(std::istreambuf_iterator<char>{included}, {});
		shaderIncludeMapping[virtualName] = text + '\n';
	}
	else
	{
		Log("Filename '{}' could not be found.", fileName);
	}
}
