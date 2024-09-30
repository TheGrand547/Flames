#pragma once
#ifndef FLAMES_LOG_H
#define FLMAES_LOG_H
#include <iostream>
#include <glew.h>
#include <source_location>
#include <string>

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
	if ((pos = func.rfind(cDecl)) != std::string::npos)
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

#ifdef _DEBUG

#define LogF(...) {printf("%s", LocationFormat().c_str()); printf(__VA_ARGS__);}
#define LogSourceF(x, ...) {printf("%s", LocationFormat(x).c_str()); printf(__VA_ARGS__);}
#define Log(...) {std::cout << LocationFormat() << __VA_ARGS__ << '\n';}
#define LogSource(x, ...) {std::cout << LocationFormat(x) << __VA_ARGS__ << '\n';}

#define Before(...) std::cout << "Before: " << __VA_ARGS__ << '\n';
#define After(...) std::cout << "After: " << __VA_ARGS__ << '\n';
#else // _DEBUG

#define Log(...) CheckError()
#define LogSource(...) CheckError()
#define LogF(...) CheckError()
#define LogSourceF(...) CheckError()
#define Before(...)
#define After(...)

#endif // _DEBUG
#endif // FLAMES_LOG_H