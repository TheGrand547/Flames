#pragma once
#ifndef FLAMES_LOG_H
#define FLMAES_LOG_H
#include <glew.h>

#ifdef _WIN64
#define FILEPATH_SLASH '\\'
#else 
#define FILEPATH_SLASH '/'
#endif // _WIN64

#define CheckError
#ifndef NDEBUG
#undef CheckError

#define __FILENAME__ (strrchr(__FILE__, FILEPATH_SLASH) ? strrchr(__FILE__, FILEPATH_SLASH) + 1 : __FILE__)
#define CheckError() CheckErrors(__LINE__, __FILENAME__, __FUNCTION__)
#endif

void CheckErrors(int line, const char *file, const char *function);

#endif // FLAMES_LOG_H