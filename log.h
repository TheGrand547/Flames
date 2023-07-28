#pragma once
#ifndef FLAMES_LOG_H
#define FLMAES_LOG_H

#ifdef _WIN64
#define FILEPATH_SLASH '\\'
#else 
#define FILEPATH_SLASH '/'
#endif // _WIN64

#ifndef OMIT_FILENAMES

#else

#endif // OMIT_FILENAMES

#ifndef NDEBUG

#define __FILENAME__ ([] (const char *file) constexpr {return (strrchr(file, FILEPATH_SLASH) ? strrchr(file, FILEPATH_SLASH) + 1 : file);}(__FILE__))
#define CheckError() CheckErrors(__LINE__, __FILENAME__, __FUNCTION__)
#define LogF(...) {printf("[%s][%s][%i] ", __FILENAME__, __FUNCTION__, __LINE__); printf(__VA_ARGS__);}
#define Log(...) {printf("[%s][%s][%i] ", __FILENAME__, __FUNCTION__, __LINE__); std::cout << __VA_ARGS__ << std::endl;}

#else // NDEBUG

#define CheckError
#define Log(...) CheckError
#define LogF(...) CheckError

#endif // NDEBUF

void CheckErrors(int line, const char *file, const char *function);

#endif // FLAMES_LOG_H