#pragma once
#ifndef FLAMES_LOG_H
#define FLMAES_LOG_H
#include <iostream>
#include <glew.h>
#include <source_location>
#include <string>
#include <format>

#ifdef _WIN64
constexpr auto FILEPATH_SLASH = '\\';
#else 
constexpr auto FILEPATH_SLASH = '/';
#endif // _WIN64

constexpr std::string LocationFormat(const std::source_location location = std::source_location::current());

#ifndef OMIT_FILENAMES

constexpr std::string LocationFormat(const std::source_location location)
{
	constexpr const char* cDecl = "__cdecl";
	constexpr const char* replaceFiller = "[0][1][2] ";
	constexpr std::size_t length = std::string(cDecl).length() + 1;
	constexpr std::size_t lineLocation = std::string(replaceFiller).find('2');
	constexpr std::size_t funcLocation = std::string(replaceFiller).find('1');
	constexpr std::size_t fileLocation = std::string(replaceFiller).find('0');


	std::string file = location.file_name();
	std::string func = location.function_name();
	std::uint_least32_t line = location.line();
	std::size_t pos{};
	if ((pos = file.rfind(FILEPATH_SLASH)) != std::string::npos)
		file = file.substr(pos + 1);
	while ((pos = func.rfind(cDecl)) != std::string::npos)
		func = func.erase(pos, length);
#ifdef CLEANER_FUNCTIONS
	if ((pos = func.find('(')) != std::string::npos)
		func = func.substr(0, pos);
#endif // CLEANER_FUNCTIONS
	std::string result = replaceFiller;
	result = result.replace(lineLocation, 1, std::to_string(line));
	result = result.replace(funcLocation, 1, func);
	result = result.replace(fileLocation, 1, file);
	return result;
}

#else // OMIT_FILENAMES

constexpr std::string LocationFormat(const std::source_location location)
{
	constexpr const char* cDecl = "__cdecl";
	constexpr const char* replaceFiller = "[0][1] ";
	constexpr std::size_t length = std::string(cDecl).length() + 1;
	constexpr std::size_t lineLocation = std::string(replaceFiller).find('1');
	constexpr std::size_t funcLocation = std::string(replaceFiller).find('0');

	std::string func = location.function_name();
	std::uint_least32_t line = location.line();
	std::size_t pos{};

	if ((pos = func.rfind(cDecl)) != std::string::npos)
		func = func.erase(pos, length);
#ifdef CLEANER_FUNCTIONS
	if ((pos = func.find('(')) != std::string::npos)
		func = func.substr(0, pos);
#endif // CLEANER_FUNCTIONS
	std::string result = replaceFiller;
	result = result.replace(lineLocation, 1, std::to_string(line));
	result = result.replace(funcLocation, 1, func);
	return result;
}

#endif // OMIT_FILENAMES

void CheckError(const std::source_location location = std::source_location::current());
void DebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);

void OutputText(const std::string_view& stringer);
void InitLog();
void CloseLog();

#define Log(...) {OutputText(LocationFormat() + std::format(__VA_ARGS__));}
#define LogSource(x, ...) {OutputText(LocationFormat(x) + std::format(__VA_ARGS__));}

#endif // FLAMES_LOG_H