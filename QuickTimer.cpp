#include "QuickTimer.h"
#include <iostream>
#include "log.h"


QuickTimer::QuickTimer(const std::string& name, const float& threshold, const std::source_location source) 
	noexcept : start(std::chrono::high_resolution_clock::now()), source(source), name((name == "") ? "QuickTimer" : name), 
	threshold(static_cast<unsigned int>(threshold * 1000))
{

}

QuickTimer::~QuickTimer() noexcept
{
	std::chrono::steady_clock::time_point current = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> ms = std::chrono::duration<float, std::chrono::milliseconds::period>(current - this->start);
	if (ms.count() > this->threshold)
	{
		LogSource(this->source, "{}: Completed in {}ms", this->name, ms.count());
	}
}
