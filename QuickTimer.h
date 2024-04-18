#pragma once
#ifndef QUICK_TIMER_H
#define QUICK_TIMER_H
#include <chrono>
#include <source_location>

class QuickTimer
{
protected:
	const std::chrono::steady_clock::time_point start;
	const std::source_location source;
	const std::string name;
public:
	QuickTimer(const std::string& named = "", const std::source_location source = std::source_location::current());
	~QuickTimer();
};

#endif // QUICK_TIMER_H
