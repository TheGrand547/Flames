#include "QuickTimer.h"
#include <iostream>
#include "log.h"


QuickTimer::QuickTimer(const std::source_location source) : start(std::chrono::high_resolution_clock::now()), source(source)
{
}

static const char * SpliceName(const char* file)
{
	return (strrchr(file, FILEPATH_SLASH) ? strrchr(file, FILEPATH_SLASH) + 1 : file); 
}

QuickTimer::~QuickTimer()
{
	std::chrono::steady_clock::time_point current = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> ms = std::chrono::duration<float, std::chrono::milliseconds::period>(current - this->start);
	std::cout << std::format("[{}][{}][{}] QuickTimer: Completed in {}ms\n", SpliceName(this->source.file_name()),
		this->source.function_name(), this->source.line(), ms.count());
}
