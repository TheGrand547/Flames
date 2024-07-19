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
	const unsigned int threshold;
public:
	QuickTimer(const std::string& named = "", const float& threshold = 0.f, const std::source_location source = std::source_location::current());
	~QuickTimer();
};

#define __QUICKTIMER(x, y) QuickTimer __##y##(x)
#define __QUICKTIMER2(x, y) __QUICKTIMER(x, y)
#define QUICKTIMER(x) __QUICKTIMER2(x, __LINE__)

#endif // QUICK_TIMER_H
