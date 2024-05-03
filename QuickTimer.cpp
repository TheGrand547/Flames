#include "QuickTimer.h"
#include <iostream>
#include "log.h"


QuickTimer::QuickTimer(const std::string& name, const std::source_location source) : start(std::chrono::high_resolution_clock::now()), source(source),
			name((name == "") ? "QuickTimer" : name)
{

}

QuickTimer::~QuickTimer()
{
	std::chrono::steady_clock::time_point current = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> ms = std::chrono::duration<float, std::chrono::milliseconds::period>(current - this->start);
	LogSource(this->source, std::format(" {}: Completed in {}ms", this->name, ms.count()));
}
