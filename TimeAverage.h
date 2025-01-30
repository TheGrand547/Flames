#pragma once
#ifndef TIMER_AVERAGE_H
#define TIMER_AVERAGE_H
#include <chrono>
#include "CircularBuffer.h"

typedef std::chrono::nanoseconds TimeDelta;

template<std::size_t WindowSize = 100, typename Delta = long long>
struct TimerAverage
{
	void Reset() noexcept
	{
		this->currentAverage = 0;
		this->lastPercentage = 0;
		this->countedTicks = 0;
		this->storedTimes.Reset();
	}

	Delta Update(const Delta& next) noexcept
	{
		Delta contribution = next / WindowSize;
		if (this->storedTimes.IsFull())
		{
			this->currentAverage -= this->storedTimes.PushPop(contribution);
		}
		else
		{
			this->storedTimes.Push(contribution);
		}
		this->currentAverage += contribution;
		return this->currentAverage;
	}

protected:
	Delta currentAverage = 0;
	CircularBuffer<Delta, WindowSize> storedTimes;
};


#endif // TIMER_AVERAGE_H