#pragma once
#ifndef FLAMES_LOG_H
#define FLMAES_LOG_H

#ifdef _WIN64
#define FILEPATH_SLASH '\\'
#else 
#define FILEPATH_SLASH '/'
#endif // _WIN64

#define CheckError
#ifndef NDEBUG
#undef CheckError

#define __FILENAME__ ([] (const char *file) constexpr {return (strrchr(file, FILEPATH_SLASH) ? strrchr(file, FILEPATH_SLASH) + 1 : file);}(__FILE__))
#define CheckError() CheckErrors(__LINE__, __FILENAME__, __FUNCTION__)
// TODO: LOG() file thingy

#endif

void CheckErrors(int line, const char *file, const char *function);

#endif // FLAMES_LOG_H