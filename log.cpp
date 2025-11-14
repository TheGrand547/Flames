#include "log.h"
#include <glew.h>
#include <iostream>
#include <fstream>

#ifdef _DEBUG
#define OutputStream std::cout
#else
static std::ofstream logOut;
#define OutputStream logOut
#endif // _DEBUG

void InitLog()
{
#ifndef _DEBUG
	// TODO Better stuff
	logOut.open("err.log");
#else
	// Don't have it flush every time a new line is written
	std::ios_base::sync_with_stdio(false);
#endif // _DEBUG
}

void CloseLog()
{
#ifndef _DEBUG
	logOut.close();
#endif // _DEBUG
}

void OutputText(const std::string_view& stringer)
{
	// Add time signature to the log file
#ifndef _DEBUG
	
#endif
	OutputStream << stringer << '\n';
}


void CheckError(const std::source_location location)
{
	GLenum e;
	while ((e = glGetError()))
	{
		Log("{} OpenGL Error: {}", LocationFormat(location), reinterpret_cast<const char*>(gluErrorString(e)));
	}
}

void DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, [[maybe_unused]] GLsizei length, const GLchar* message,
	[[maybe_unused]] const void* userParam)
{
	// FROM https://gist.github.com/liam-middlebrook/c52b069e4be2d87a6d2f
	const char* _source;
	const char* _type;
	const char* _severity;

	switch (source) {
	case GL_DEBUG_SOURCE_API: _source = "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM: _source = "WINDOW SYSTEM"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: _source = "SHADER COMPILER"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY: _source = "THIRD PARTY"; break;
	case GL_DEBUG_SOURCE_APPLICATION: _source = "APPLICATION"; break;
	case GL_DEBUG_SOURCE_OTHER: _source = "UNKNOWN"; break;
	default: _source = "UNKNOWN"; break;
	}

	switch (type) {
	case GL_DEBUG_TYPE_ERROR: _type = "ERROR"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:_type = "DEPRECATED BEHAVIOR"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:_type = "UDEFINED BEHAVIOR"; break;
	case GL_DEBUG_TYPE_PORTABILITY:_type = "PORTABILITY"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:_type = "PERFORMANCE"; break;
	case GL_DEBUG_TYPE_OTHER:_type = "OTHER"; break;
	case GL_DEBUG_TYPE_MARKER:_type = "MARKER"; break;
	default:_type = "UNKNOWN"; break;
	}

	switch (severity) {
	case GL_DEBUG_SEVERITY_HIGH:_severity = "HIGH"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:_severity = "MEDIUM"; break;
	case GL_DEBUG_SEVERITY_LOW:_severity = "LOW"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION:_severity = "NOTIFICATION"; break;
	default:_severity = "UNKNOWN"; break;
	}

	OutputText(std::format("{}: {} of {} severity, raised from {}: {}", id, _type, _severity, _source, message));
}
